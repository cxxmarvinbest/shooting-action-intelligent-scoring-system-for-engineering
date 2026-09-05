# -*- coding: utf-8 -*-
"""
方案 A vs B vs 离线 letterbox 的精度/耗时 A/B 对比（test/ab_compare）
=====================================================================
目标：验证方案 B（等比 640x360 + 补黑边）是否在不损失耗时的前提下，
     把方案 A（暴力拉伸 640x640）丢失的检测/姿态精度补回来。

对比对象（仅预处理方式不同，检测模型 / 姿态模型完全相同）：
  A. 暴力拉伸   ：resize(1920x1080 -> 640x640)
  B. 等比+补黑边：resize(1920x1080 -> 640x360) + copyMakeBorder 上下补 140
  L. letterbox（基准）：长边 640 + 黑边（离线评分既有口径）

评价指标（均为核心）：
  1. player 检测框 IoU（相对 letterbox 基准，越接近 1 越准）
  2. player 置信度
  3. 关键点召回率（各自 player 框抠 ROI 后，可见关键点数 / 17）
  4. Python 侧预处理耗时（ms）

说明：
  - 本脚本是【离线模拟】，测的是 Python 侧预处理耗时，不能反映 RGA 硬件的
    零 CPU 缩放优势；真实链路的耗时差异主要是方案 B 多一次 copyMakeBorder
    （约 0.5~1ms），而 NPU 检测耗时三者完全一致（输入均为 640x640）。
  - 姿态 crop 统一从原图 1920x1080 抠取（三种方案框均为原图归一化坐标），
    以隔离"预处理方式"这一个变量。

用法（RK3588 板端）：
  python test/regression/ab_compare.py
"""
import sys
from pathlib import Path
# 获取当前脚本所在test文件夹的【父目录】=项目根目录，加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import time

import cv2
import numpy as np

from vision_algorithm.common.rknn_infer import letterbox_to_model
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer

VIDEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "test_videos", "left_rtsp.mp4")

PLAYER_CLS = 0
BALL_CLS = 1


# ------------------------------------------------------------------
# 三种预处理
# ------------------------------------------------------------------
def preprocess_A(f):
    t0 = time.perf_counter()
    canvas = cv2.resize(f, (640, 640), interpolation=cv2.INTER_LINEAR)
    return canvas, (time.perf_counter() - t0) * 1000.0


def preprocess_B(f):
    t0 = time.perf_counter()
    det360 = cv2.resize(f, (640, 360), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(det360, 140, 140, 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas, (time.perf_counter() - t0) * 1000.0


def preprocess_L(f):
    t0 = time.perf_counter()
    canvas, _, _, _ = letterbox_to_model(f, 640, 640, pad_color=(0, 0, 0))
    return canvas, (time.perf_counter() - t0) * 1000.0


# ------------------------------------------------------------------
# 归一化（640x640 画布框 -> 原图归一化 [0,1]）
# ------------------------------------------------------------------
def norm_A(box):
    """方案 A：640x640 拉伸，直接除以 640。"""
    return (box[0] / 640.0, box[1] / 640.0, box[2] / 640.0, box[3] / 640.0)


def norm_BL(box):
    """方案 B / L：等比 640x360 内容在画布 y∈[140,500]，反算回 640x360 再归一化。"""
    return (box[0] / 640.0, (box[1] - 140) / 360.0,
            box[2] / 640.0, (box[3] - 140) / 360.0)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def crop_roi(f, norm_box):
    """按原图归一化框从原图抠 ROI；无效返回 None。"""
    H, W = f.shape[:2]
    x1 = max(0, int(norm_box[0] * W))
    y1 = max(0, int(norm_box[1] * H))
    x2 = min(W - 1, int(norm_box[2] * W))
    y2 = min(H - 1, int(norm_box[3] * H))
    if x1 >= x2 or y1 >= y2:
        return None
    return f[y1:y2, x1:x2]


def kpt_visible(poses):
    """返回最优姿态的可见关键点数（x 或 y > 0）。"""
    if not poses:
        return 0
    best = max(poses, key=lambda p: (p['box'][2] - p['box'][0])
               * (p['box'][3] - p['box'][1]))
    k = best['kpts']
    return int(((k[:, 0] > 0) | (k[:, 1] > 0)).sum())


def pick(dets, cls_id):
    """取指定类别中面积最大的框 + 置信度（canvas 坐标）。"""
    cand = [d for d in dets if d['cls'] == cls_id]
    if not cand:
        return None, None
    best = max(cand, key=lambda d: (d['box'][2] - d['box'][0])
               * (d['box'][3] - d['box'][1]))
    return best['box'], best['conf']


def main():
    analyzer = VideoAnalyzer()
    analyzer.load_models()
    det = analyzer.det_model
    pose = analyzer.pose_model

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"打开视频失败: {VIDEO}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"已打开: {VIDEO}  total≈{total}，目标采集 30 帧含 player 的帧\n")

    acc = {"A_iou": [], "B_iou": [],
           "A_conf": [], "B_conf": [], "L_conf": [],
           "A_kpt": [], "B_kpt": [], "L_kpt": [],
           "A_ms": [], "B_ms": [], "L_ms": []}

    collected = 0
    frame_idx = 0
    while collected < 30:
        ok, f = cap.read()
        if not ok:
            break
        if frame_idx % 30 != 0:
            frame_idx += 1
            continue

        cA, msA = preprocess_A(f)
        cB, msB = preprocess_B(f)
        cL, msL = preprocess_L(f)
        dA = det.detect_on_canvas(cA)
        dB = det.detect_on_canvas(cB)
        dL = det.detect_on_canvas(cL)

        boxA, confA = pick(dA, PLAYER_CLS)
        boxB, confB = pick(dB, PLAYER_CLS)
        boxL, confL = pick(dL, PLAYER_CLS)

        # 只统计「letterbox 基准检出 player」的帧
        if boxL is None:
            frame_idx += 1
            continue

        nA = norm_A(boxA) if boxA else None
        nB = norm_BL(boxB) if boxB else None
        nL = norm_BL(boxL)

        acc["A_iou"].append(iou(nA, nL) if nA else 0.0)
        acc["B_iou"].append(iou(nB, nL) if nB else 0.0)
        acc["A_conf"].append(confA if confA else 0.0)
        acc["B_conf"].append(confB if confB else 0.0)
        acc["L_conf"].append(confL)

        # 关键点召回率：各自 player 框抠 ROI（无框/无效 ROI 计 0）
        for key, nb in (("A", nA), ("B", nB), ("L", nL)):
            if nb is None:
                acc[f"{key}_kpt"].append(0)
                continue
            roi = crop_roi(f, nb)
            if roi is None or roi.shape[0] < 8 or roi.shape[1] < 8:
                acc[f"{key}_kpt"].append(0)
                continue
            acc[f"{key}_kpt"].append(kpt_visible(pose.detect_crop(roi)))

        acc["A_ms"].append(msA)
        acc["B_ms"].append(msB)
        acc["L_ms"].append(msL)

        collected += 1
        print(f"[帧 {frame_idx}] A_IoU={acc['A_iou'][-1]:.3f} "
              f"B_IoU={acc['B_iou'][-1]:.3f} "
              f"conf(A/B/L)={acc['A_conf'][-1]:.2f}/{acc['B_conf'][-1]:.2f}/{acc['L_conf'][-1]:.2f} "
              f"kpt(A/B/L)={acc['A_kpt'][-1]}/{acc['B_kpt'][-1]}/{acc['L_kpt'][-1]}/17 "
              f"ms(A/B/L)={msA:.2f}/{msB:.2f}/{msL:.2f}")
        frame_idx += 1

    cap.release()
    analyzer.release_models()

    if collected == 0:
        print("未采集到含 player 的帧")
        return

    print("\n" + "=" * 60)
    print(f"汇总（共 {collected} 帧）")
    print("=" * 60)
    print("player 框 IoU（相对 letterbox 基准，越接近 1 越准）:")
    print(f"  方案A(拉伸)      : {np.mean(acc['A_iou']):.3f}")
    print(f"  方案B(等比+补边) : {np.mean(acc['B_iou']):.3f}")
    print("player 置信度均值:")
    print(f"  方案A : {np.mean(acc['A_conf']):.3f}")
    print(f"  方案B : {np.mean(acc['B_conf']):.3f}")
    print(f"  基准L : {np.mean(acc['L_conf']):.3f}")
    print("关键点召回率均值（可见点数/17）:")
    print(f"  方案A : {np.mean(acc['A_kpt']):.2f}")
    print(f"  方案B : {np.mean(acc['B_kpt']):.2f}")
    print(f"  基准L : {np.mean(acc['L_kpt']):.2f}")
    print("预处理耗时均值（ms，Python 侧模拟，不含 RGA 硬件收益）:")
    print(f"  方案A : {np.mean(acc['A_ms']):.2f}")
    print(f"  方案B : {np.mean(acc['B_ms']):.2f}")
    print(f"  基准L : {np.mean(acc['L_ms']):.2f}")


if __name__ == "__main__":
    main()
