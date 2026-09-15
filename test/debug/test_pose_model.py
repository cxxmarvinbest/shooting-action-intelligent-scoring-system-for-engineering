# -*- coding: utf-8 -*-
"""
姿态模型单图单元测试（test/debug/test_pose_model）
=====================================================
对一张图跑 yolov8n-pose（4 输出，flatten 关键点），打印检测到的人与 17 个关键点，
并把关键点正确映射回「原图坐标系」，保存骨架结果图。

关键点：本脚本【直接用 outputs[3] 解析关键点】，不依赖业务 pose_model 的旧后处理。
  - pose 模型 4 输出：outputs[0..2] = 3 尺度 box+cls；outputs[3] = flatten 关键点。
  - 旧「3 输出 / 9 输出」后处理会把关键点取空导致全 0，故必须用第 4 路输出。

两种模式（对应业务真实链路的隔离排查）：
  ① 整图模式（默认）      ：整张图直接作为 pose 输入（等价 crop=整图，偏移=0）
  ② crop 模式（--crop）    ：先按检测框 x1,y1,x2,y2 抠 ROI，再 letterbox 送入 pose，
                            模拟业务「检测框 -> 原图 crop roi -> resize 送入 pose」链路。
关键点坐标链路（与业务 video_analyzer 一致）：
  pose 输入图坐标 --反letterbox((x-dw)/scale)--> crop 图坐标 --加crop偏移--> 原图坐标。

用法（RK3588 板上，项目根目录）：
  python3 test/debug/test_pose_model.py /abs/path/test.jpg
  python3 test/debug/test_pose_model.py /abs/path/test.jpg --crop 100,200,600,900
  python3 test/debug/test_pose_model.py /abs/path/test.jpg \
      --model /abs/path/weights/yolov8n-pose.rknn --model-w 224 --model-h 480
"""
import sys
from pathlib import Path
# 获取脚本所在往上两层，即项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from vision_algorithm.common.rknn_infer import preprocess_to_input  # noqa: E402
from vision_algorithm.pose.pose_feature import SKELETON_CONNECTIONS  # noqa: E402

# pose4out.py 与本脚本同目录，直接同目录导入（避免 test/ 无 __init__.py 时
# `from test.debug.pose4out` 被标准库 test 包遮蔽导致 ModuleNotFoundError）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from pose4out import parse_pose4  # noqa: E402

KP_NAMES = ["鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
            "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"]


def _load_rknn(model_path, core_mask):
    """加载 RKNN 模型并 init_runtime，返回 rknn 实例。失败抛 RuntimeError。"""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        raise RuntimeError("未安装 rknn-toolkit-lite2，请在 RK3588 板上运行本脚本")

    if not os.path.exists(model_path):
        raise RuntimeError(f"模型文件不存在: {model_path}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败 ret={ret}（模型加载错误）: {model_path}")
    try:
        if core_mask is not None:
            ret = rknn.init_runtime(core_mask=core_mask)
        else:
            ret = rknn.init_runtime()
    except TypeError:
        ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime 失败 ret={ret}（NPU 资源异常）")
    return rknn


def _infer_pose(rknn, crop, model_w, model_h):
    """crop 图 -> letterbox -> 推理 -> 返回 (outputs, scale, dw, dh)。"""
    img, scale, dw, dh = preprocess_to_input(crop, model_w, model_h)
    outputs = rknn.inference(inputs=[img], data_format="nhwc")
    return outputs, scale, dw, dh


def _map_to_orig(kpts_input, scale, dw, dh, crop_x1, crop_y1):
    """关键点从 pose 输入图坐标 -> 原图坐标（反 letterbox + 加 crop 偏移）。"""
    k = kpts_input.copy()
    k[:, 0] = (k[:, 0] - dw) / scale + crop_x1
    k[:, 1] = (k[:, 1] - dh) / scale + crop_y1
    return k


def draw_skeleton(frame, results):
    """在 frame 上原地画骨架 + 关键点（results 坐标为 frame 坐标系）。"""
    for r in results:
        bx1, by1, bx2, by2 = [int(v) for v in r['box']]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
        cv2.putText(frame, f"person {r['conf']:.2f}", (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        kpts = r['kpts']
        kpt_conf = r.get('kpt_conf')
        for a, b in SKELETON_CONNECTIONS:
            pa, pb = kpts[a], kpts[b]
            if pa[0] > 0 and pb[0] > 0:
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (255, 150, 0), 2)
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


def _diagnose_all_zero(results, diag):
    """关键点全 0 时的根因诊断（区分 4 类问题）。"""
    lines = []
    if len(diag.get("output_shapes", [])) < 4:
        lines.append("  [后处理解析错误] 模型输出不足 4 路，缺少 flatten 关键点输出（outputs[3]）")
    if "kpts_error" in diag:
        lines.append(f"  [后处理解析错误] {diag['kpts_error']}")
    if "error" in diag:
        lines.append(f"  [后处理解析错误] 关键点形状无法识别: {diag['error']}")
    krange = diag.get("conf_range")
    if krange is not None and abs(krange[1]) < 1e-3:
        lines.append("  [后处理解析错误] 关键点 conf 通道全 0（导出/量化丢关键点置信度）")
    if not results:
        lines.append("  [画面无有效人体] box 无检测（conf 低于阈值，或画面无清晰人体）")
        lines.append("    建议：降低 --conf，或换一张含完整人体、光照清晰的图重试")
    if not lines:
        lines.append("  [未知原因] 见上方 shape/数值范围诊断输出")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="姿态模型单图单元测试（4 输出）")
    ap.add_argument("image", help="输入图片绝对路径")
    ap.add_argument("--model", default=None, help="pose rknn 绝对路径（默认 Config.POSE_RKNN_PATH）")
    ap.add_argument("--model-w", type=int, default=None, help="pose 输入宽（默认 Config.POSE_MODEL_W）")
    ap.add_argument("--model-h", type=int, default=None, help="pose 输入高（默认 Config.POSE_MODEL_H）")
    ap.add_argument("--crop", default=None, help="检测框 x1,y1,x2,y2（原图坐标，模拟 crop-roi 链路）")
    ap.add_argument("--conf", type=float, default=None, help="人体框置信度阈值（覆盖默认）")
    ap.add_argument("--kpt", type=float, default=None, help="关键点置信度阈值（覆盖默认）")
    ap.add_argument("--out", default=None, help="结果图输出路径")
    ap.add_argument("--json", default=None, help="JSON 落盘目录（默认 data/output；传 '' 关闭）")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"[预处理错误] 图片不存在: {args.image}")
        sys.exit(1)

    model_path = args.model or Config.POSE_RKNN_PATH
    model_w = args.model_w if args.model_w is not None else Config.POSE_MODEL_W
    model_h = args.model_h if args.model_h is not None else Config.POSE_MODEL_H
    conf = args.conf if args.conf is not None else Config.POSE_CONF_THRES
    kpt_thres = args.kpt if args.kpt is not None else Config.POSE_KPT_CONF_THRES
    nms = Config.POSE_NMS_THRES
    core_mask = Config.get("NPU_CORE_MASK", 7)

    print("=" * 68)
    print("姿态模型单图单元测试（4 输出 flatten 关键点）")
    print("=" * 68)
    print(f"模型      : {model_path}")
    print(f"输入尺寸  : {model_w}x{model_h}")
    print(f"阈值      : conf={conf} kpt={kpt_thres} nms={nms}")

    # 1) 读图（预处理错误捕获）
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[预处理错误] 读图失败: {args.image}")
        sys.exit(1)
    print(f"图片尺寸  : {frame.shape[1]}x{frame.shape[0]} (WxH)")

    # 2) crop 模式：先抠 ROI 复现业务链路；整图模式：crop=整图
    crop = frame
    crop_x1 = crop_y1 = 0
    if args.crop:
        try:
            x1, y1, x2, y2 = [int(v) for v in args.crop.split(",")]
        except Exception:
            print(f"[预处理错误] --crop 格式非法: {args.crop}（应为 x1,y1,x2,y2）")
            sys.exit(1)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        if x1 >= x2 or y1 >= y2:
            print(f"[预处理错误] 检测框非法: ({x1},{y1})-({x2},{y2})")
            sys.exit(1)
        crop = frame[y1:y2, x1:x2]
        crop_x1, crop_y1 = x1, y1
        print(f"抠图 ROI  : ({x1},{y1})-({x2},{y2})，尺寸 {crop.shape[1]}x{crop.shape[0]}")

    # 3) 加载模型 + 推理（模型加载错误捕获）
    try:
        rknn = _load_rknn(model_path, core_mask)
    except RuntimeError as e:
        print(f"[模型加载错误] {e}")
        sys.exit(1)

    try:
        outputs, scale, dw, dh = _infer_pose(rknn, crop, model_w, model_h)
    except Exception as e:
        print(f"[推理失败] {type(e).__name__}: {e}")
        rknn.release()
        sys.exit(1)

    # 4) 4 输出解析（用 outputs[3] 解析关键点）
    results, diag = parse_pose4(outputs, model_w, model_h, conf, nms)
    print(f"\n模型输出路数: {diag.get('n_outputs')}  各输出 shape: {diag.get('output_shapes')}")
    if "kpts_shape" in diag:
        print(f"关键点张量 shape: {diag['kpts_shape']}  "
              f"x范围{diag.get('x_range')}  y范围{diag.get('y_range')}  "
              f"conf范围{diag.get('conf_range')}")
    for w in diag.get("warnings", []):
        print(f"  [注意] {w}")

    # 5) 关键点 conf 阈值 + 坐标映射回原图
    mapped_results = []
    for r in results:
        kpts_input = np.hstack([r['kpts'], r['kpt_conf'][:, None]])  # (17,3)
        low = r['kpt_conf'] < kpt_thres
        kpts_input[low, 0] = 0.0
        kpts_input[low, 1] = 0.0
        kpts_orig = _map_to_orig(kpts_input[:, :2], scale, dw, dh, crop_x1, crop_y1)
        # 越界保护（原图范围）
        kpts_orig[:, 0] = np.clip(kpts_orig[:, 0], 0, frame.shape[1] - 1)
        kpts_orig[:, 1] = np.clip(kpts_orig[:, 1], 0, frame.shape[0] - 1)
        bx = _map_to_orig(np.array([[r['box'][0], r['box'][1]],
                                    [r['box'][2], r['box'][3]]], dtype=np.float32),
                          scale, dw, dh, crop_x1, crop_y1)
        mapped_results.append({
            'box': (int(bx[0, 0]), int(bx[0, 1]), int(bx[1, 0]), int(bx[1, 1])),
            'conf': r['conf'],
            'kpts': kpts_orig,
            'kpt_conf': r['kpt_conf'],
        })

    # 6) 打印 n/17
    print(f"\n检测到人体数: {len(mapped_results)}")
    summary = []
    for i, r in enumerate(mapped_results):
        kpts = r['kpts']
        visible = int((np.asarray(kpts)[:, 0] > 0).sum())
        print(f"  人[{i}]: box={r['box']} conf={r['conf']:.3f}  可见关键点: {visible}/17")
        for j in range(17):
            x, y = kpts[j]
            flag = "OK " if (x > 0 and y > 0) else " 零"
            print(f"         {flag} [{j:2d}] {KP_NAMES[j]:3s} ({x:6.1f}, {y:6.1f})")
        summary.append({"idx": i, "box": list(r['box']), "conf": r['conf'],
                        "visible_kpts": visible, "total_kpts": 17})

    # 7) 全 0 诊断
    if not mapped_results or all(int((r['kpts'][:, 0] > 0).sum()) == 0 for r in mapped_results):
        print("\n[诊断] 关键点全 0 / 未检出人体，可能原因：")
        print(_diagnose_all_zero(mapped_results, diag))

    # 8) 保存骨架图
    draw_frame = frame.copy()
    draw_skeleton(draw_frame, mapped_results)
    out_path = args.out or os.path.join(PROJECT_ROOT, "pose_result.jpg")
    cv2.imwrite(out_path, draw_frame)
    print(f"\n结果图已保存: {out_path}")

    # 9) JSON 落盘
    if args.json != "":
        out_dir = args.json or os.path.join(PROJECT_ROOT, "data", "output")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        jp = os.path.join(out_dir, f"pose_model_report_{ts}.json")
        payload = {
            "generated_at": ts, "image": args.image, "model": model_path,
            "model_w": model_w, "model_h": model_h,
            "n_person": len(mapped_results),
            "persons": summary,
            "diag": {k: v for k, v in diag.items() if not isinstance(v, np.ndarray)},
        }
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"报告已落盘: {jp}")

    rknn.release()


if __name__ == "__main__":
    main()
