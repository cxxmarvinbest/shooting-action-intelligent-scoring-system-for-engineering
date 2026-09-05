# -*- coding: utf-8 -*-
"""
单独测试姿态估计模型：对一张图片跑 yolov8n-pose，打印检测到的人与 17 个关键点
（含每个关键点置信度），并保存画了骨架/关键点的结果图。

用途：排查「姿态点 0/17」到底是【姿态模型本身】的问题，还是【检测→抠图→姿态】通路的问题。
  - 若本脚本在完整清晰的人像上能出关键点 → 模型没问题，问题在检测通路的 crop/letterbox/坐标；
  - 若本脚本也出 0 个人 / 0 关键点 → 模型或模型输入（尺寸/布局/量化）有问题。

用法（RK3588 板上，项目根目录）：
    python3 test/debug/test_pose_model.py <图片路径> [--model weights/yolov8n-pose.rknn] [--out 结果图]
    python3 test/debug/test_pose_model.py <图片路径> --crop x1,y1,x2,y2   # 先抠 ROI 再测（复现检测通路）
    python3 test/debug/test_pose_model.py <图片路径> --conf 0.1 --kpt 0.3  # 临时调阈值排查
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from common.logger import setup_logger  # noqa: E402
from vision_algorithm.common.rknn_infer import preprocess_to_input  # noqa: E402
from vision_algorithm.pose.pose_model import RKNNPoseModel  # noqa: E402
from vision_algorithm.pose.pose_feature import SKELETON_CONNECTIONS  # noqa: E402

KP_NAMES = ["鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
            "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"]


def draw_skeleton(frame, results):
    """在 frame 上原地画骨架 + 关键点（results 坐标为 frame 坐标系）。"""
    for r in results:
        bx1, by1, bx2, by2 = [int(v) for v in r['box']]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
        cv2.putText(frame, f"person {r['conf']:.2f}", (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        kpts = r['kpts']          # (17, 2)
        kpt_conf = r.get('kpt_conf')  # (17,)
        # 骨架连线
        for a, b in SKELETON_CONNECTIONS:
            pa, pb = kpts[a], kpts[b]
            if pa[0] > 0 and pb[0] > 0:
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (255, 150, 0), 2)
        # 关键点
        for i, p in enumerate(kpts):
            if p[0] <= 0:
                continue
            color = (0, 255, 255) if i <= 4 else (0, 0, 255)
            r_ = 3 if i <= 4 else 4
            cv2.circle(frame, (int(p[0]), int(p[1])), r_, color, -1)
            if kpt_conf is not None:
                cv2.putText(frame, f"{i}:{kpt_conf[i]:.2f}",
                            (int(p[0]) + 4, int(p[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="姿态估计模型单图测试")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--model", default=None, help="pose rknn 路径（默认 Config.POSE_RKNN_PATH）")
    parser.add_argument("--crop", default=None, help="先抠图再测，格式 x1,y1,x2,y2")
    parser.add_argument("--conf", type=float, default=None, help="人体框置信度阈值（覆盖默认）")
    parser.add_argument("--kpt", type=float, default=None, help="关键点置信度阈值（覆盖默认）")
    parser.add_argument("--out", default=None, help="结果图输出路径")
    parser.add_argument("--dump", action="store_true", help="只 dump 模型原始输出张量形状（排查关键点格式）")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"图片不存在: {args.image}")
        sys.exit(1)

    setup_logger(os.path.join(PROJECT_ROOT, "logs"))

    model_path = args.model or Config.POSE_RKNN_PATH
    print(f"姿态模型: {model_path}")
    print(f"输入尺寸: {Config.POSE_MODEL_W}x{Config.POSE_MODEL_H}")

    conf = args.conf if args.conf is not None else Config.POSE_CONF_THRES
    kpt = args.kpt if args.kpt is not None else Config.POSE_KPT_CONF_THRES
    model = RKNNPoseModel(model_path, conf_thres=conf, kpt_conf_thres=kpt,
                          nms_thres=Config.POSE_NMS_THRES,
                          model_w=Config.POSE_MODEL_W, model_h=Config.POSE_MODEL_H)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"读图失败: {args.image}")
        sys.exit(1)
    print(f"图片尺寸: {frame.shape[1]}x{frame.shape[0]} (WxH)")

    # 原始输出 dump 模式：直接推理并打印每个输出张量的形状/范围，判断关键点输出格式
    if args.dump:
        img, _scale, _dw, _dh = preprocess_to_input(
            frame, Config.POSE_MODEL_W, Config.POSE_MODEL_H)
        outputs = model.infer_input(img)
        print("\n原始输出张量:")
        for i, o in enumerate(outputs):
            if o is None:
                print(f"  output[{i}]: None")
                continue
            arr = np.asarray(o)
            print(f"  output[{i}]: shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min():.4f} max={arr.max():.4f}")
        model.release()
        return

    # 抠图模式（复现检测→抠图→姿态通路）
    crop = None
    if args.crop:
        x1, y1, x2, y2 = [int(v) for v in args.crop.split(",")]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        crop = frame[y1:y2, x1:x2]
        print(f"抠图 ROI: ({x1},{y1})-({x2},{y2})，尺寸 {crop.shape[1]}x{crop.shape[0]}")
        results = model.detect_crop(crop)
    else:
        results = model.detect(frame)

    print(f"\n检测到人体数: {len(results)}")
    for i, r in enumerate(results):
        print(f"  人[{i}]: box={tuple(int(v) for v in r['box'])} conf={r['conf']:.3f}")
        kpts = r['kpts']
        kpt_conf = r.get('kpt_conf')
        visible = int((np.asarray(kpts)[:, 0] > 0).sum())
        print(f"         可见关键点: {visible}/17")
        for j in range(17):
            x, y = kpts[j]
            c = kpt_conf[j] if kpt_conf is not None else -1
            flag = "OK " if (x > 0 and y > 0) else "  零"
            print(f"         {flag} [{j:2d}] {KP_NAMES[j]:3s} ({x:6.1f}, {y:6.1f}) conf={c:.3f}")

    # 画图保存
    draw_frame = (crop.copy() if crop is not None else frame.copy())
    draw_skeleton(draw_frame, results)
    out_path = args.out or os.path.join(PROJECT_ROOT, "pose_result.jpg")
    cv2.imwrite(out_path, draw_frame)
    print(f"\n结果图已保存: {out_path}")

    model.release()


if __name__ == "__main__":
    main()
