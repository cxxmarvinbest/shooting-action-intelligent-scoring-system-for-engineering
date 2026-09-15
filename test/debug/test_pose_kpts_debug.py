# -*- coding: utf-8 -*-
"""
姿态关键点全 0 根因诊断（test/debug/test_pose_kpts_debug）
=============================================================
直接读 yolov8n-pose **4 输出**的原始张量，定位「关键点全 0」到底卡在哪一环：

  问题0) outputs[3]（flatten 关键点）里到底有没有有效人体关键点
         —— 全局扫描 conf 最高的锚点，并直接把高 conf 关键点画到 320 画布上，
            不依赖 box 检测框，纯看关键点张量自身是否已含人体。
  问题1) 3 路 box 输出在哪个尺度、哪个锚点检出人体（conf 最高框）
  问题2) 最高置信框在 flatten 关键点里取到什么关键点值（x/y/conf）
  问题3) 尺度拼接顺序双假设验证：
         flatten 关键点的 2100 个锚点，究竟是「大尺度(40x40)在前」还是
         「小尺度(10x10)在前」。两种假设分别取关键点，conf 高的那个即正确顺序。

关键点：本脚本【直接用 outputs[3] 解析关键点】。旧 3 输出后处理会把关键点取空
导致全 0，故必须看第 4 路输出。

用法（RK3588 板上，项目根目录）：
  python3 test/debug/test_pose_kpts_debug.py /abs/path/test.jpg
  python3 test/debug/test_pose_kpts_debug.py /abs/path/test.jpg \
      --model /abs/path/weights/yolov8n_pose_int8_320.rknn --model-w 320 --model-h 320
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
from vision_algorithm.common.rknn_infer import preprocess_to_input  # noqa: E402

# pose4out.py 与本脚本同目录，直接同目录导入（避免 test/ 无 __init__.py 时
# `from test.debug.pose4out` 被标准库 test 包遮蔽导致 ModuleNotFoundError）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from pose4out import normalize_kpts_flatten, parse_box_outputs  # noqa: E402


def _scales_from_box_outputs(outputs):
    """从 3 路 box 输出里取每个尺度的 (h, w)，按面积降序返回。

    返回 list[(h, w)]，含所有 3 个尺度（不管是否检出框）。
    """
    hws = []
    for out in outputs[:3]:
        arr = np.asarray(out)
        if arr.ndim != 4:
            continue
        # 判定布局：65/64 通道在最后一维 => NHWC，否则 NCHW
        if arr.shape[3] in (64, 65):
            h, w = arr.shape[1], arr.shape[2]
        else:
            h, w = arr.shape[2], arr.shape[3]
        if (h, w) not in hws:
            hws.append((h, w))
    return sorted(hws, key=lambda hw: -hw[0] * hw[1])


def _scan_kpts_peaks(kpts):
    """扫描 flatten 关键点 [17,3,N]，返回每个锚点 max conf 的 top-K 全局索引。"""
    maxc = kpts[:, 2, :].max(0)          # (N,) 每个锚点 17 点取最大 conf
    order = np.argsort(-maxc)            # conf 降序的锚点索引
    return maxc, order


def _fetch_kpts_by_hypothesis(kpts, flat_idx, scale_hw, scale_order, small_first):
    """按给定的尺度顺序假设，取某个检测框对应的关键点 (17,3)。

    scale_order: 按面积降序的尺度列表 [(h,w), ...]（即 box 输出顺序）。
    small_first=True 时，flatten 关键点的拼接顺序取「小尺度在前」（反转 scale_order）。
    """
    order = scale_order[::-1] if small_first else scale_order
    off = 0
    for hw in order:
        if hw == scale_hw:
            break
        off += hw[0] * hw[1]
    gi = off + flat_idx
    if gi < kpts.shape[2]:
        return kpts[:, :, gi], gi
    return None, gi


def main():
    ap = argparse.ArgumentParser(description="姿态 4 输出关键点全 0 根因诊断")
    ap.add_argument("image", help="输入图片绝对路径")
    ap.add_argument("--model", default=None, help="pose rknn 绝对路径")
    ap.add_argument("--model-w", type=int, default=None, help="pose 输入宽")
    ap.add_argument("--model-h", type=int, default=None, help="pose 输入高")
    ap.add_argument("--out", default=None, help="关键点可视化图输出路径（默认项目根 kpts_peaks.jpg）")
    args = ap.parse_args()

    model_path = args.model or Config.POSE_RKNN_PATH
    model_w = args.model_w if args.model_w is not None else Config.POSE_MODEL_W
    model_h = args.model_h if args.model_h is not None else Config.POSE_MODEL_H

    if not os.path.exists(args.image):
        print(f"[预处理错误] 图片不存在: {args.image}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[模型加载错误] 模型不存在: {model_path}")
        sys.exit(1)

    print(f"姿态模型: {model_path}")
    print(f"输入尺寸: {model_w}x{model_h}")

    frame = cv2.imread(args.image)
    if frame is None:
        print("[预处理错误] 读图失败")
        sys.exit(1)
    print(f"图片尺寸: {frame.shape[1]}x{frame.shape[0]} (WxH)")

    # 加载 rknn + 推理（整图直接作为 pose 输入）
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("[模型加载错误] 未安装 rknn-toolkit-lite2，请在 RK3588 板上运行")
        sys.exit(1)

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"[模型加载错误] load_rknn 失败 ret={ret}")
        sys.exit(1)
    try:
        ret = rknn.init_runtime(core_mask=Config.get("NPU_CORE_MASK", 7))
    except TypeError:
        ret = rknn.init_runtime()
    if ret != 0:
        print(f"[模型加载错误] init_runtime 失败 ret={ret}")
        sys.exit(1)

    img, scale, dw, dh = preprocess_to_input(frame, model_w, model_h)
    outputs = rknn.inference(inputs=[img], data_format="nhwc")

    # ── 总览：4 输出 shape ──
    print("\n【总览】模型输出路数 =", len(outputs))
    for i, o in enumerate(outputs):
        arr = np.asarray(o)
        print(f"  outputs[{i}]: shape={arr.shape} dtype={arr.dtype} "
              f"min={float(arr.min()):.4f} max={float(arr.max()):.4f}")
    if len(outputs) < 4:
        print("\n[后处理解析错误] 输出不足 4 路，缺少 flatten 关键点输出（outputs[3]）。")
        print("  请确认导出为 flatten=True 的 4 输出结构。")
        rknn.release()
        sys.exit(1)

    # ── 问题0：outputs[3] 关键点张量全局扫描 ──
    print("\n【问题0】outputs[3] flatten 关键点全局扫描:")
    kpts, kdiag = normalize_kpts_flatten(outputs[3])
    if kpts is None:
        print(f"  [后处理解析错误] {kdiag.get('error')}")
        rknn.release()
        sys.exit(1)
    print(f"  规范化后形状 (17,3,N): {kdiag.get('kpts_shape')}")
    N = kpts.shape[2]
    print(f"  锚点总数 N={N}")
    print(f"  x   通道: min={kdiag['x_range'][0]:.1f} max={kdiag['x_range'][1]:.1f}")
    print(f"  y   通道: min={kdiag['y_range'][0]:.1f} max={kdiag['y_range'][1]:.1f}")
    print(f"  conf通道: min={kdiag['conf_range'][0]:.4f} max={kdiag['conf_range'][1]:.4f}")
    if kdiag.get("conf_sigmoid_applied"):
        print("  [注意] conf 超出 [0,1]，已补 sigmoid（说明导出未对关键点 conf 做 sigmoid）")

    maxc, order = _scan_kpts_peaks(kpts)
    print(f"  conf>0.3 的锚点数: {int((maxc > 0.3).sum())}/{N}")
    print(f"  conf>0.5 的锚点数: {int((maxc > 0.5).sum())}/{N}")
    print("  conf 最高的 top-10 锚点（global_idx / 最高conf / 可见点数>0.3）:")
    for gi in order[:10]:
        vals = kpts[:, :, gi]
        n_vis = int((vals[:, 2] > 0.3).sum())
        print(f"    gi={gi:5d}  max_conf={maxc[gi]:.3f}  可见点={n_vis:2d}/17")

    # ── 问题1：3 路 box 输出的检测框 ──
    print("\n【问题1】3 路 box 输出各尺度最大置信度:")
    dets, bdiag = parse_box_outputs(outputs, model_w, model_h, conf_thres=0.001)
    for s in bdiag.get("scales", []):
        print(f"  scale(h,w)={s.get('hw')}  检出数={s.get('n_det')}")

    scale_order = _scales_from_box_outputs(outputs)
    print(f"  box 输出尺度顺序(面积降序): {scale_order}")

    if dets:
        best = max(dets, key=lambda d: d['conf'])
        print(f"  → 最高置信度框: box={best['box']} conf={best['conf']:.4f} "
              f"scale={best['scale_hw']} flat_idx={best['flat_idx']}")

        # ── 问题3：双尺度顺序假设验证 ──
        print("\n【问题3】尺度拼接顺序双假设验证（哪种假设下关键点 conf 高，即正确顺序）:")
        for small_first, tag in [(False, "大尺度(40x40)在前"), (True, "小尺度(10x10)在前")]:
            kk, gi = _fetch_kpts_by_hypothesis(
                kpts, best['flat_idx'], best['scale_hw'], scale_order, small_first)
            if kk is None:
                print(f"  [{tag}] 假设下 global_idx={gi} 越界（N={N}）")
                continue
            conf = kk[:, 2]
            n_vis = int((conf > 0.3).sum())
            mean_c = float(conf.mean())
            print(f"  [{tag}] global_idx={gi}: conf均值={mean_c:.3f} "
                  f"可见点(>0.3)={n_vis}/17")
            print(f"        conf: {np.array2string(conf, precision=2)}")
            print(f"        前5点(x,y,conf): "
                  + ", ".join(f"({kk[j,0]:6.1f},{kk[j,1]:6.1f},{kk[j,2]:.2f})"
                              for j in range(5)))
    else:
        print("  [画面无有效人体] 3 路 box 输出均未检出 conf>0.001 的人体框（画面可能无清晰人体）")

    # ── 可视化：把高 conf 关键点直接画到 320 画布（不依赖 box 对应）──
    canvas = np.zeros((model_h, model_w, 3), dtype=np.uint8)
    drawn = 0
    for gi in range(N):
        if maxc[gi] <= 0.3:
            continue
        vals = kpts[:, :, gi]
        conf = vals[:, 2]
        for j in range(17):
            if conf[j] <= 0.3:
                continue
            x, y = vals[j, 0], vals[j, 1]
            if 0 <= x < model_w and 0 <= y < model_h:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 255), -1)
                drawn += 1
    out_path = args.out or os.path.join(PROJECT_ROOT, "kpts_peaks.jpg")
    cv2.imwrite(out_path, canvas)
    print(f"\n关键点可视化图已保存: {out_path}（画布 {model_w}x{model_h}，"
          f"共绘制 {drawn} 个 conf>0.3 关键点；若成人体轮廓说明关键点张量本身正常，"
          f"问题在 box→anchor 对应）")

    rknn.release()


if __name__ == "__main__":
    main()
