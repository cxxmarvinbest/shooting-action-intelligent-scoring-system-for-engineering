# -*- coding: utf-8 -*-
"""
C++ 引擎 vs rknn_lite 引擎的检测 A/B 对比（test/ab_cpp_engine）
=================================================================
目标：验证 C++ 检测引擎（rknn_yolov8 pybind11）替换 Python 后处理时，
     精度不劣化、篮球小目标召回可接受、耗时下降。

对比对象（检测模型完全相同 = best_int8.rknn，仅后处理实现不同）：
  - cpp      ：CppDetModel.detect_360（C++ 后处理，阈值硬编码 0.25）
  - rknn_lite：RKNNDetModel.detect_on_canvas（Python numpy 后处理，
               player 0.10 / ball 0.05 双阈值）

评价指标（坐标统一反算到 640x360 输入图坐标系，对齐报告 3.1 表）：
  1. player 召回率（lite 检出 player 的帧中，cpp 也检出的比例）
  2. ball   召回率（lite 检出 ball 的帧中，cpp 也检出的比例）—— 核心风险点
     （cpp 单阈值 0.25 对篮球小目标可能漏检）
  3. player 框 IoU（两引擎都检出的帧）
  4. ball   框 IoU（两引擎都检出的帧）
  5. cpp 单帧耗时 / lite 单帧耗时（含各自预处理）

用法（RK3588 板端，.so 已部署到 vision_algorithm/detection/）：
  python test/regression/ab_cpp_engine.py                  # 精度 + 耗时对比（both）
  python test/regression/ab_cpp_engine.py --frames 60      # 采集帧数
  python test/regression/ab_cpp_engine.py --mode cpp       # 单测 cpp 引擎耗时
  python test/regression/ab_cpp_engine.py --mode lite      # 单测 lite 引擎耗时
  python test/regression/ab_cpp_engine.py --video /path/to/x.mp4

说明：
  - both 模式下两个 RKNN 上下文并存，NPU 分时调度，耗时仅供参考；
    纯耗时请用 --mode cpp / --mode lite 单独测，避免核心抢占失真。
  - 结果同时落盘 JSON（data/output/report_ab_cpp_engine_*.json）便于回填报告 3.1。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 获取当前脚本所在 test 文件夹的【父目录】=项目根目录，加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from config import Config
from vision_algorithm.detection.det_model import RKNNDetModel
from vision_algorithm.detection.cpp_det_model import CppDetModel

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VIDEO = os.path.join(PROJ_ROOT, "test", "test_videos", "left_rtsp.mp4")

PLAYER_CLS = 0
BALL_CLS = 1
DET_W, DET_H = 640, 360   # RGA 等比缩放输出（方案 B）
MODEL_SZ = 640

# 达标阈值（与《测试方案》3.1 一致）
THRESH_P_IOU = 0.9
THRESH_P_RECALL = 95.0
THRESH_B_RECALL = 80.0


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def pick(dets, cls_id):
    """取指定类别中面积最大的框 + 置信度。"""
    cand = [d for d in dets if d['cls'] == cls_id]
    if not cand:
        return None, None
    best = max(cand, key=lambda d: (d['box'][2] - d['box'][0])
               * (d['box'][3] - d['box'][1]))
    return best['box'], best['conf']


def has_cls(dets, cls_id):
    return any(d['cls'] == cls_id for d in dets)


def letterbox_lite(det360):
    """rknn_lite 引擎：det360 上下补黑边到 640x640（与 _extract_frame_metrics_norm 一致）。"""
    dh = (MODEL_SZ - DET_H) // 2
    dw = (MODEL_SZ - DET_W) // 2
    canvas = cv2.copyMakeBorder(det360, dh, dh, dw, dw,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas, dw, dh


def lite_box_to_360(box, dw, dh):
    """640x640 画布坐标 -> 640x360 坐标（反算补边）。"""
    return (box[0] - dw, box[1] - dh, box[2] - dw, box[3] - dh)


def load_det360(f):
    """原图 -> 等比缩放 640x360（模拟 RGA 输出，无畸变）。"""
    return cv2.resize(f, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)


def run_single(model, frames_target, is_cpp, video):
    """单引擎纯耗时测试（避免双上下文 NPU 抢占）。返回统计 dict。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"打开视频失败: {video}")
        return None
    times = []
    frame_idx = 0
    while len(times) < frames_target:
        ok, f = cap.read()
        if not ok:
            break
        if frame_idx % 10 != 0:
            frame_idx += 1
            continue
        det360 = load_det360(f)
        if is_cpp:
            t0 = time.perf_counter()
            model.detect_360(det360)
            times.append((time.perf_counter() - t0) * 1000.0)
        else:
            canvas, dw, dh = letterbox_lite(det360)
            t0 = time.perf_counter()
            model.detect_on_canvas(canvas)
            times.append((time.perf_counter() - t0) * 1000.0)
        frame_idx += 1
    cap.release()
    if not times:
        return None
    stat = {
        "engine": "cpp" if is_cpp else "rknn_lite",
        "n": len(times),
        "mean_ms": round(float(np.mean(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
    }
    print(f"单引擎耗时均值: {stat['mean_ms']:.2f}ms  "
          f"(min={stat['min_ms']:.2f}, max={stat['max_ms']:.2f}, n={stat['n']})")
    return stat


def run_both(cpp, lite, frames_target, video):
    """精度 + 耗时对比（同一帧分别喂两个引擎）。返回汇总 dict。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"打开视频失败: {video}")
        return None

    acc = {"p_iou": [], "b_iou": [],
           "p_conf_cpp": [], "p_conf_lite": [],
           "b_conf_cpp": [], "b_conf_lite": [],
           "cpp_ms": [], "lite_ms": []}
    lite_player_frames = 0
    cpp_player_match = 0
    lite_ball_frames = 0
    cpp_ball_match = 0

    frame_idx = 0
    collected = 0
    while collected < frames_target:
        ok, f = cap.read()
        if not ok:
            break
        if frame_idx % 10 != 0:
            frame_idx += 1
            continue
        det360 = load_det360(f)

        # cpp：直接喂 640x360，内部 letterbox
        t0 = time.perf_counter()
        dets_cpp = cpp.detect_360(det360)
        ms_cpp = (time.perf_counter() - t0) * 1000.0

        # lite：补黑边到 640x640 再检测
        canvas, dw, dh = letterbox_lite(det360)
        t0 = time.perf_counter()
        dets_lite = lite.detect_on_canvas(canvas)
        ms_lite = (time.perf_counter() - t0) * 1000.0

        # player / ball 召回统计（只对 lite 检出的帧评估）
        if has_cls(dets_lite, PLAYER_CLS):
            lite_player_frames += 1
            if has_cls(dets_cpp, PLAYER_CLS):
                cpp_player_match += 1
        if has_cls(dets_lite, BALL_CLS):
            lite_ball_frames += 1
            if has_cls(dets_cpp, BALL_CLS):
                cpp_ball_match += 1

        # IoU / 置信度（两引擎都检出时才对比）
        box_cpp_p, conf_cpp_p = pick(dets_cpp, PLAYER_CLS)
        box_lite_p, conf_lite_p = pick(dets_lite, PLAYER_CLS)
        if box_cpp_p and box_lite_p:
            acc["p_iou"].append(iou(box_cpp_p,
                                    lite_box_to_360(box_lite_p, dw, dh)))
            acc["p_conf_cpp"].append(conf_cpp_p)
            acc["p_conf_lite"].append(conf_lite_p)

        box_cpp_b, conf_cpp_b = pick(dets_cpp, BALL_CLS)
        box_lite_b, conf_lite_b = pick(dets_lite, BALL_CLS)
        if box_cpp_b and box_lite_b:
            acc["b_iou"].append(iou(box_cpp_b,
                                    lite_box_to_360(box_lite_b, dw, dh)))
            acc["b_conf_cpp"].append(conf_cpp_b)
            acc["b_conf_lite"].append(conf_lite_b)

        acc["cpp_ms"].append(ms_cpp)
        acc["lite_ms"].append(ms_lite)

        collected += 1
        p_iou = acc["p_iou"][-1] if box_cpp_p and box_lite_p else float("nan")
        print(f"[帧 {frame_idx}] p_IoU={p_iou:.3f}  "
              f"cpp={len(dets_cpp)}目标 lite={len(dets_lite)}目标  "
              f"ms(cpp/lite)={ms_cpp:.2f}/{ms_lite:.2f}")
        frame_idx += 1

    cap.release()

    p_recall = (cpp_player_match / lite_player_frames * 100.0
                if lite_player_frames else float("nan"))
    b_recall = (cpp_ball_match / lite_ball_frames * 100.0
                if lite_ball_frames else float("nan"))
    p_iou_mean = float(np.mean(acc["p_iou"])) if acc["p_iou"] else float("nan")
    b_iou_mean = float(np.mean(acc["b_iou"])) if acc["b_iou"] else float("nan")
    cpp_ms = float(np.mean(acc["cpp_ms"]))
    lite_ms = float(np.mean(acc["lite_ms"]))

    print("\n" + "=" * 64)
    print(f"C++ 引擎 vs rknn_lite 引擎 A/B 汇总（采集 {collected} 帧）")
    print("=" * 64)
    print("召回率（lite 检出为基准，cpp 也检出的比例）:")
    print(f"  player 召回 : {p_recall:.1f}%  ({cpp_player_match}/{lite_player_frames})")
    print(f"  ball   召回 : {b_recall:.1f}%  ({cpp_ball_match}/{lite_ball_frames})  ← 核心风险点")
    print("框 IoU（两引擎都检出时，越接近 1 越准）:")
    print(f"  player IoU : {p_iou_mean:.3f}" if not np.isnan(p_iou_mean) else "  player IoU : 无共检帧")
    print(f"  ball   IoU : {b_iou_mean:.3f}" if not np.isnan(b_iou_mean) else "  ball   IoU : 无共检帧")
    print("置信度均值（cpp / lite）:")
    print(f"  player : {np.mean(acc['p_conf_cpp']):.3f} / {np.mean(acc['p_conf_lite']):.3f}"
          if acc["p_conf_cpp"] else "  player : 无共检帧")
    print(f"  ball   : {np.mean(acc['b_conf_cpp']):.3f} / {np.mean(acc['b_conf_lite']):.3f}"
          if acc["b_conf_cpp"] else "  ball   : 无共检帧")
    print("单帧检测耗时均值（含预处理，both 模式受 NPU 抢占影响，仅供参考）:")
    print(f"  cpp      : {cpp_ms:.2f}ms")
    print(f"  rknn_lite: {lite_ms:.2f}ms")

    # 达标判定
    print("\n达标判定:")
    checks = {
        "player_iou": bool(not np.isnan(p_iou_mean) and p_iou_mean >= THRESH_P_IOU),
        "player_recall": bool(p_recall >= THRESH_P_RECALL),
        "ball_recall": bool(b_recall >= THRESH_B_RECALL),
    }
    ok = True
    if checks["player_iou"]:
        print(f"  [PASS] player IoU = {p_iou_mean:.3f} ≥ {THRESH_P_IOU}")
    else:
        print(f"  [FAIL] player IoU 不达标或未共检（当前 {p_iou_mean:.3f}）")
        ok = False
    if checks["player_recall"]:
        print(f"  [PASS] player 召回 = {p_recall:.1f}% ≥ {THRESH_P_RECALL}%")
    else:
        print(f"  [WARN] player 召回 = {p_recall:.1f}%（cpp 阈值 0.25 可能漏低置信球员）")
        ok = False
    if checks["ball_recall"]:
        print(f"  [PASS] ball 召回 = {b_recall:.1f}% ≥ {THRESH_B_RECALL}%")
    else:
        print(f"  [WARN] ball 召回 = {b_recall:.1f}%（篮球小目标在 0.25 阈值下漏检，需评估）")
        ok = False
    conclusion = "通过，可切 DET_ENGINE=cpp" if ok else \
                 "未完全达标，见上方 WARN/FAIL，暂不建议切 cpp"
    print("\n结论: " + conclusion)

    return {
        "frames": collected,
        "player_recall_pct": round(float(p_recall), 1),
        "player_recall_n": f"{cpp_player_match}/{lite_player_frames}",
        "ball_recall_pct": round(float(b_recall), 1),
        "ball_recall_n": f"{cpp_ball_match}/{lite_ball_frames}",
        "player_iou": None if np.isnan(p_iou_mean) else round(p_iou_mean, 3),
        "ball_iou": None if np.isnan(b_iou_mean) else round(b_iou_mean, 3),
        "cpp_ms": round(cpp_ms, 2),
        "lite_ms": round(lite_ms, 2),
        "checks": checks,
        "passed": ok,
        "conclusion": conclusion,
    }


def _dump_report(report, video):
    """落盘 JSON 到 data/output，便于回填报告 3.1。"""
    out_dir = os.path.join(PROJ_ROOT, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"report_ab_cpp_engine_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": ts, "video": video, **report},
                  f, ensure_ascii=False, indent=2)
    print(f"\n报告已落盘: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="C++ vs rknn_lite 检测引擎 A/B 对比（报告 3.1）")
    ap.add_argument("--mode", default="both", choices=["both", "cpp", "lite"],
                    help="对比模式（默认 both）")
    ap.add_argument("--frames", type=int, default=30, help="采集帧数（默认 30）")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="测试视频路径")
    args = ap.parse_args()

    video = args.video
    core_mask = Config.get("NPU_CORE_MASK", 7)
    print(f"检测模型: {Config.DET_RKNN_PATH}")
    print(f"测试视频: {video}")
    print(f"当前 DET_ENGINE 配置: {Config.get('DET_ENGINE', 'rknn_lite')}")
    print("【风险提示】cpp 引擎 conf 阈值硬编码 0.25，无法对篮球单独放宽；"
          "篮球是小目标，召回是核心风险点，务必关注下方 ball 召回是否 ≥80%。\n")

    cpp = CppDetModel(Config.DET_RKNN_PATH, core_mask=core_mask)

    if args.mode == "cpp":
        print(f"单测 cpp 引擎耗时（{args.frames} 帧）...")
        stat = run_single(cpp, args.frames, is_cpp=True, video=video)
        cpp.release()
        if stat:
            _dump_report({"mode": "cpp", "latency": stat}, video)
        return

    lite = RKNNDetModel(
        Config.DET_RKNN_PATH,
        conf_thres=Config.DET_CONF_THRES,
        nms_thres=Config.DET_NMS_THRES,
        ball_conf_thres=Config.get("DET_BALL_CONF_THRES", 0.30),
        model_w=640, model_h=640, core_mask=core_mask)

    if args.mode == "lite":
        print(f"单测 lite 引擎耗时（{args.frames} 帧）...")
        stat = run_single(lite, args.frames, is_cpp=False, video=video)
        lite.release()
        cpp.release()
        if stat:
            _dump_report({"mode": "lite", "latency": stat}, video)
        return

    report = run_both(cpp, lite, args.frames, video)
    lite.release()
    cpp.release()
    if report:
        _dump_report({"mode": "both", **report}, video)


if __name__ == "__main__":
    main()
