# -*- coding: utf-8 -*-
"""
诊断 yolov8n-pose「人体框能检出，但 17 个关键点全 0」的根因。

直接读模型原始输出张量，回答三个问题：
  问题1) 关键点张量里到底有没有非零的置信度（模型是否真输出了关键点）
  问题2) 检测框（box+cls）在哪个尺度、哪个锚点上
  问题3) 检测框锚点在两种「尺度拼接顺序」假设下，各取到什么关键点值

用法（RK3588 板上，项目根目录）:
    python3 test/debug/test_pose_kpts_debug.py test.jpg [--model weights/yolov8n-pose.rknn]
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
from vision_algorithm.common.rknn_infer import (  # noqa: E402
    preprocess_to_input, sigmoid, as_nchw)
from vision_algorithm.pose.pose_model import RKNNPoseModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--model", default=None, help="pose rknn 路径")
    args = parser.parse_args()

    setup_logger(os.path.join(PROJECT_ROOT, "logs"))
    model_path = args.model or Config.POSE_RKNN_PATH
    print(f"姿态模型: {model_path}")
    print(f"输入尺寸: {Config.POSE_MODEL_W}x{Config.POSE_MODEL_H}")

    model = RKNNPoseModel(model_path, conf_thres=0.01, kpt_conf_thres=0.0,
                          nms_thres=Config.POSE_NMS_THRES,
                          model_w=Config.POSE_MODEL_W, model_h=Config.POSE_MODEL_H)

    frame = cv2.imread(args.image)
    if frame is None:
        print("读图失败")
        sys.exit(1)
    print(f"图片尺寸: {frame.shape[1]}x{frame.shape[0]} (WxH)")

    img, _s, _dw, _dh = preprocess_to_input(frame, model.model_w, model.model_h)
    outputs = model.infer_input(img)

    # ── 分离关键点张量与各尺度 box+cls ──
    kpts_tensor = None
    for out in outputs:
        if out.ndim == 4 and 17 in out.shape[1:] and 3 in out.shape[1:]:
            kpts_tensor = out
            break

    is_nhwc = None
    for out in outputs:
        if out.ndim == 4 and (65 in out.shape[1:] or 64 in out.shape[1:]):
            is_nhwc = (out.shape[3] in (64, 65))
            break

    boxcls = {}
    for out in outputs:
        if out.ndim != 4:
            continue
        if 17 in out.shape[1:] and 3 in out.shape[1:]:
            continue
        o = as_nchw(out, is_nhwc)
        if o.shape[1] == 65:
            hw = (o.shape[2], o.shape[3])
            boxcls[hw] = (o[:, :64, :, :], o[:, 64:, :, :])

    if kpts_tensor is None:
        print("未找到 (17,3,N) 关键点张量！")
        model.release()
        sys.exit(1)

    print(f"\n关键点张量形状: {tuple(kpts_tensor.shape)}")
    k = kpts_tensor[0]  # (17, 3, N)
    N = k.shape[2]
    print(f"锚点总数 N={N}")

    # ── 问题1：关键点张量里到底有没有非零值 ──
    print("\n【问题1】关键点张量 3 个通道的全局统计（假设通道顺序 = x, y, conf）:")
    print(f"  x  通道: max={k[:, 0, :].max():.1f}  min={k[:, 0, :].min():.1f}")
    print(f"  y  通道: max={k[:, 1, :].max():.1f}  min={k[:, 1, :].min():.1f}")
    print(f"  conf通道: max={k[:, 2, :].max():.4f}  mean={k[:, 2, :].mean():.4f}")
    maxc = k[:, 2, :].max(0)  # 每个锚点的最大 conf（17 关键点取最大）
    print(f"  conf>0.3 的锚点数: {int((maxc > 0.3).sum())}/{N}")
    print(f"  conf>0.5 的锚点数: {int((maxc > 0.5).sum())}/{N}")

    # ── 问题2：检测框在哪个尺度/锚点 ──
    scales = sorted(boxcls.keys(), key=lambda hw: -hw[0] * hw[1])
    print("\n【问题2】各尺度检测框最大置信度:")
    scale_conf = []
    for hw in scales:
        _, cls = boxcls[hw]
        conf = sigmoid(cls.astype(np.float32))[:, 0].flatten()
        f = int(np.argmax(conf))
        scale_conf.append((hw, f, float(conf[f])))
        print(f"  scale(h,w)={hw}: max_conf={conf[f]:.4f} @ flat_idx={f}")
    best = max(scale_conf, key=lambda x: x[2])
    best_hw, best_f, best_conf = best
    print(f"  → 最高置信度框: scale={best_hw}, flat_idx={best_f}, conf={best_conf:.4f}")

    # ── 问题3：两种顺序假设下取关键点 ──
    def global_idx(order, hw, f):
        off = 0
        for s in order:
            if s == hw:
                return off + f
            off += s[0] * s[1]
        return -1

    orders = [
        ("降序(大尺度在前)[60,30,15]", sorted(scales, key=lambda hw: -hw[0] * hw[1])),
        ("升序(小尺度在前)[15,30,60]", sorted(scales, key=lambda hw: hw[0] * hw[1])),
    ]

    print("\n【问题3】检测框锚点在两种顺序假设下取到的关键点:")
    for name, order in orders:
        gi = global_idx(order, best_hw, best_f)
        vals = k[:, :, gi]  # (17, 3)
        conf = vals[:, 2]
        print(f"\n  假设 {name}: 全局锚点索引={gi}")
        print(f"    conf: {np.array2string(conf, precision=2)}")
        print(f"    conf>0.3 的个数: {int((conf > 0.3).sum())}/17")
        print(f"    前5点 (x, y, conf):")
        for j in range(5):
            print(f"      kp{j}: x={vals[j, 0]:6.1f} y={vals[j, 1]:6.1f} conf={vals[j, 2]:.3f}")

    # ── 全局最高 conf 锚点分布（交叉验证顺序） ──
    top = np.argsort(maxc)[-10:][::-1]
    print("\n【全局】conf 最高的 10 个锚点索引及其 conf:")
    for idx in top:
        d = (">> 降序假设尺度: " +
             ("60" if idx < 3600 else "30" if idx < 4500 else "15"))
        a = ("  | 升序假设尺度: " +
             ("15" if idx < 225 else "30" if idx < 1125 else "60"))
        print(f"    anchor={idx:5d} conf={maxc[idx]:.4f} {d}{a}")

    model.release()


if __name__ == "__main__":
    main()
