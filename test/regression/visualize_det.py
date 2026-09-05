# -*- coding: utf-8 -*-
"""
抽帧画框可视化：cpp vs rknn_lite 检测框对比（test/visualize_det）
=================================================================
用途：A/B 对比发现「帧 130 后 cpp 检出数 > lite」后，把指定帧的
      cpp 与 lite 检测框画出来左右并排，人工判断 cpp 多检的框
      是真实目标（player/ball）还是误检。

用法（RK3588 板端，.so 已部署到 vision_algorithm/detection/）：
  python test/regression/visualize_det.py                      # 默认抽 130/180/250
  python test/regression/visualize_det.py --frames 130,180,250
  python test/regression/visualize_det.py --out debug_imgs     # 指定输出目录

输出：
  - 每帧一张左右并排图（左 cpp / 右 lite），文件名 debug_f<idx>_cpp_vs_lite.jpg
  - 终端同步打印两引擎每个框的 cls/conf/box，即使不看图也能从数值初判

坐标口径：
  - 两列均统一反算到 640x360 输入图坐标系（lite 由 640x640 画布 -dw/-dh 反算），
    再放大 SCALE 倍显示，保证左右坐标对齐、可直接比对。
  - 框颜色：player 绿 / ball 蓝；标注「类别 置信度」。
"""
import sys
from pathlib import Path
# 获取当前脚本所在test文件夹的【父目录】=项目根目录，加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import sys

import cv2
import numpy as np

from config import Config
from vision_algorithm.detection.det_model import RKNNDetModel
from vision_algorithm.detection.cpp_det_model import CppDetModel

VIDEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "test_videos", "left_rtsp.mp4")

CLS_NAMES = {0: "player", 1: "ball"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}   # BGR：player 绿 / ball 蓝
DET_W, DET_H = 640, 360
MODEL_SZ = 640
SCALE = 2                                   # 显示放大倍数（640x360 -> 1280x720）


def load_det360(f):
    """原图 -> 等比缩放 640x360（模拟 RGA 输出，无畸变）。"""
    return cv2.resize(f, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)


def letterbox_lite(det360):
    """rknn_lite 引擎：det360 上下补黑边到 640x640（与 ab_cpp_engine 一致）。"""
    dh = (MODEL_SZ - DET_H) // 2
    dw = (MODEL_SZ - DET_W) // 2
    canvas = cv2.copyMakeBorder(det360, dh, dh, dw, dw,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas, dw, dh


def lite_box_to_360(box, dw, dh):
    """640x640 画布坐标 -> 640x360 坐标（反算补边）。"""
    return (box[0] - dw, box[1] - dh, box[2] - dw, box[3] - dh)


def draw_dets(img, dets, scale):
    """在放大后的图上画检测框（坐标按 scale 缩放，并 clip 到图像范围）。"""
    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = [int(v * scale) for v in d['box']]
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
        cls = d['cls']; conf = d['conf']
        color = COLORS.get(cls, (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLS_NAMES.get(cls, cls)} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def add_title(img, text):
    """顶部加黑条标题，区分左右两列引擎。"""
    bar = np.zeros((44, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    argv = sys.argv[1:]
    frames = [130, 180, 250]
    out_dir = "."
    if "--frames" in argv:
        frames = [int(x) for x in argv[argv.index("--frames") + 1].split(",")]
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
    os.makedirs(out_dir, exist_ok=True)

    core_mask = Config.get("NPU_CORE_MASK", 7)
    print(f"检测模型: {Config.DET_RKNN_PATH}")

    cpp = CppDetModel(Config.DET_RKNN_PATH, core_mask=core_mask)
    lite = RKNNDetModel(
        Config.DET_RKNN_PATH,
        conf_thres=Config.DET_CONF_THRES,
        nms_thres=Config.DET_NMS_THRES,
        ball_conf_thres=Config.get("DET_BALL_CONF_THRES", 0.30),
        model_w=MODEL_SZ, model_h=MODEL_SZ, core_mask=core_mask)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"打开视频失败: {VIDEO}")
        return

    targets = set(frames)
    frame_idx = 0
    while targets:
        ok, f = cap.read()
        if not ok:
            print(f"视频读完，未找到帧: {sorted(targets)}")
            break
        if frame_idx not in targets:
            frame_idx += 1
            continue

        det360 = load_det360(f)

        # cpp：直接喂 640x360，内部 letterbox，返回输入图坐标
        dets_cpp = cpp.detect_360(det360)

        # lite：补黑边到 640x640 再检测，反算回 640x360
        canvas, dw, dh = letterbox_lite(det360)
        dets_lite = lite.detect_on_canvas(canvas)
        dets_lite_360 = [dict(box=lite_box_to_360(d['box'], dw, dh),
                              cls=d['cls'], conf=d['conf']) for d in dets_lite]

        img_cpp = cv2.resize(det360, (DET_W * SCALE, DET_H * SCALE))
        img_lite = img_cpp.copy()
        draw_dets(img_cpp, dets_cpp, SCALE)
        draw_dets(img_lite, dets_lite_360, SCALE)

        img_cpp = add_title(img_cpp, f"cpp   frame {frame_idx}   {len(dets_cpp)} det")
        img_lite = add_title(img_lite, f"lite  frame {frame_idx}   {len(dets_lite)} det")

        panel = np.hstack([img_cpp, img_lite])
        out_path = os.path.join(out_dir, f"debug_f{frame_idx}_cpp_vs_lite.jpg")
        cv2.imwrite(out_path, panel)
        print(f"[帧 {frame_idx}] cpp={len(dets_cpp)}目标 lite={len(dets_lite)}目标 -> {out_path}")
        for d in dets_cpp:
            print(f"    cpp  : {CLS_NAMES.get(d['cls'], d['cls'])} "
                  f"conf={d['conf']:.2f} box={d['box']}")
        for d in dets_lite_360:
            print(f"    lite : {CLS_NAMES.get(d['cls'], d['cls'])} "
                  f"conf={d['conf']:.2f} box={d['box']}")

        targets.discard(frame_idx)
        frame_idx += 1

    cap.release()
    cpp.release()
    lite.release()
    print("完成")


if __name__ == "__main__":
    main()
