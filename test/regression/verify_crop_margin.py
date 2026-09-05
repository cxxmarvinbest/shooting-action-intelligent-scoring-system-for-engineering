# -*- coding: utf-8 -*-
"""
T2-4 副作用验证：扩边 margin -> letterbox 320 后人变小 -> 关键点精度损失量化
（同时覆盖 T2-3 收益端：贴边丢失点数量）

================================================================================
物理本质（要验证的副作用链）：
  player_box -> _crop_with_margin(box, img, margin)  # 四边向外扩 margin
             -> crop = box + 2*margin
             -> preprocess_to_input(crop, 320, 320)  # 保持宽高比 letterbox
                  scale = min(320/crop_w, 320/crop_h)
             -> 人体在 320 画布中的实际分辨率 = box_h * scale
  margin↑ -> crop↑ -> scale↓ -> 人体被缩小 -> 关键点定位精度↓

  损失与人体尺度强相关（box_h 越小，越敏感）：
    box_h=300（近） margin=10 -> human_px = 300*320/320 = 300（降 6.3%）
    box_h=100（远） margin=10 -> human_px = 100*320/120 = 267（降 16.7%）
  故远场小框是副作用重灾区。本脚本对远场视频（最坏情况）跑出的结论可直接外推。

隔离变量：同一帧、同一 player_box，只变 margin ∈ {0,10,20,30}，排除检测抖动。
参照基准：margin=0 作「准真值」（crop 最小、人体占比最大、分辨率天花板）。
比较口径：只统计「margin=0 与 margin=M 两边都可见」的关键点，避免把
          「margin=0 贴边丢了、margin>0 才检出的点」误算成坐标偏移。

用法（RK3588 板端，需 rknn-toolkit-lite2 与 rknn_yolov8 .so 已部署）：
  python test/regression/verify_crop_margin.py                                  # 默认 50 帧、4 档
  python test/regression/verify_crop_margin.py --video /path/to/far_field.mp4   # 指定远场视频
  python test/regression/verify_crop_margin.py --frames 50 --margins 0,10,20,30

判定阈值（T2-4 已锁定）：
  归一化偏移均值 < 1.0% box_h   通过；1.0~2.0% 关注；> 2.0% 阻塞
  归一化偏移 P95  < 2.0% box_h   通过；2.0~4.0% 关注；> 4.0% 阻塞
  人体分辨率损失  < 10%          通过；10~20% 关注；> 20% 阻塞
================================================================================
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# 项目根目录加入 sys.path（脚本位于 test/，需能 import config 与 vision_algorithm）
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import Config
from vision_algorithm.common.rknn_infer import letterbox_to_model
from vision_algorithm.detection.cpp_det_model import CppDetModel
from vision_algorithm.detection.target_selector import TargetSelector
from vision_algorithm.pose.pose_model import RKNNPoseModel
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer

DEFAULT_VIDEO = os.path.join(PROJ_ROOT, "test", "left_side_basketball.mp4")
NUM_KPTS = 17

# 人体尺度分层（box_h 为原图像素，1920x1080 坐标）
SCALE_BINS = [
    ("远场小框 (box_h < 150)",   lambda h: h < 150),
    ("中景框   (150~300)",        lambda h: 150 <= h < 300),
    ("近场大框 (box_h >= 300)",   lambda h: h >= 300),
]


def p95(a):
    """第 95 百分位（空数组返回 nan）。"""
    return float(np.percentile(a, 95)) if len(a) else float("nan")


def detect_player_box(det_model, frame):
    """离线链路检测部分：letterbox -> detect_on_canvas -> 主球员框反算回原图坐标。

    返回 (box_orig, scale, dh)；无 player 返回 (None, None, None)。
    box_orig 为原图坐标 (x1,y1,x2,y2) 浮点。
    """
    canvas, scale, dw, dh = letterbox_to_model(
        frame, det_model.model_w, det_model.model_h, pad_color=(0, 0, 0))
    dets = det_model.detect_on_canvas(canvas)
    box_640 = TargetSelector.select_main_player_box(
        dets, Config.get("DET_PLAYER_CLS_ID", 0))
    if box_640 is None:
        return None, None, None
    x1 = (box_640[0] - dw) / scale
    y1 = (box_640[1] - dh) / scale
    x2 = (box_640[2] - dw) / scale
    y2 = (box_640[3] - dh) / scale
    return (x1, y1, x2, y2), scale, dh


def run_pose_margin(pose_model, frame, box, margin):
    """对同一 player_box 按指定 margin 抠图 -> 姿态 -> 关键点回映射原图。

    返回 dict：kpts(17,2 原图坐标) / visible(17 bool) / scale(letterbox 缩放) /
               crop_w / crop_h；抠图为空或未检出人返回 None。
    """
    crop, cx, cy = VideoAnalyzer._crop_with_margin(box, frame, margin)
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    crop_h, crop_w = crop.shape[:2]
    scale = min(Config.POSE_MODEL_W / crop_w, Config.POSE_MODEL_H / crop_h)

    poses = pose_model.detect_crop(crop)
    if not poses:
        return None
    best = max(poses, key=lambda p: (p['box'][2] - p['box'][0])
               * (p['box'][3] - p['box'][1]))
    kpts_crop = best['kpts'].copy()          # (17,2) 裁剪图坐标
    visible = (kpts_crop[:, 0] > 0) | (kpts_crop[:, 1] > 0)
    kpts_orig = np.zeros_like(kpts_crop, dtype=np.float32)
    kpts_orig[visible, 0] = kpts_crop[visible, 0] + cx
    kpts_orig[visible, 1] = kpts_crop[visible, 1] + cy
    return {'kpts': kpts_orig, 'visible': visible, 'scale': scale,
            'crop_w': crop_w, 'crop_h': crop_h}


def main():
    ap = argparse.ArgumentParser(description="扩边 margin 对姿态关键点精度的副作用验证")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="测试视频路径")
    ap.add_argument("--frames", type=int, default=50, help="采样帧数（默认 50）")
    ap.add_argument("--margins", default="0,10,20,30",
                    help="margin 档位，逗号分隔（默认 0,10,20,30）")
    ap.add_argument("--out", default=None, help="报告落盘目录（默认 data/output）")
    args = ap.parse_args()

    margins = [int(x) for x in args.margins.split(",") if x.strip() != ""]
    if 0 not in margins:
        margins = [0] + margins   # 0 是准真值基准，必须存在
    margins = sorted(set(margins))
    kpt_thres = Config.get("POSE_KPT_CONF_THRES", 0.3)

    print("=" * 68)
    print("T2-4 扩边 margin 副作用验证（远场视频 = 最坏情况）")
    print("=" * 68)
    print(f"视频       : {args.video}")
    print(f"采样帧数   : {args.frames}")
    print(f"margin 档位: {margins}  (0=准真值基准)")
    print(f"检测引擎   : {Config.get('DET_ENGINE', 'cpp')}")
    print(f"姿态模型   : {Config.POSE_RKNN_PATH}  (input {Config.POSE_MODEL_W}x{Config.POSE_MODEL_H})")
    print(f"关键点阈值 : kpt_conf > {kpt_thres}\n")

    # 加载模型（复用真实链路同款实例化参数）
    core_mask = Config.get("NPU_CORE_MASK", 7)
    det_model = CppDetModel(Config.DET_RKNN_PATH, core_mask=core_mask)
    pose_model = RKNNPoseModel(
        Config.POSE_RKNN_PATH,
        conf_thres=Config.POSE_CONF_THRES,
        nms_thres=Config.POSE_NMS_THRES,
        kpt_conf_thres=kpt_thres,
        model_w=Config.get("POSE_MODEL_W", 320),
        model_h=Config.get("POSE_MODEL_H", 320),
        core_mask=core_mask)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FAIL] 打开视频失败: {args.video}")
        det_model.release()
        pose_model.release()
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, total // max(1, args.frames))
    print(f"视频信息   : total≈{total} 帧, fps={fps:.1f}, 采样步长={stride}\n")

    # 采样：每隔 stride 帧处理一帧，检测到 player 才计入，采满 args.frames 帧
    records = []          # 每条: {'idx','box_h','per_margin': {margin: run_pose_margin 结果}}
    frame_idx = 0
    t_start = time.perf_counter()
    while len(records) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        box_orig, _, _ = detect_player_box(det_model, frame)
        if box_orig is None:
            frame_idx += 1
            continue

        x1, y1, x2, y2 = box_orig
        box_h = float(y2 - y1)
        if box_h <= 0:
            frame_idx += 1
            continue

        per_margin = {}
        for m in margins:
            per_margin[m] = run_pose_margin(pose_model, frame, box_orig, m)
        records.append({'idx': frame_idx, 'box_h': box_h, 'per_margin': per_margin})
        frame_idx += 1

        if len(records) % 10 == 0:
            print(f"  已采 {len(records)}/{args.frames} 帧（最近一帧 box_h={box_h:.0f}px）")

    cap.release()
    elapsed = time.perf_counter() - t_start
    n_valid = len(records)
    print(f"\n采样完成: 有效帧 {n_valid}/{args.frames}，耗时 {elapsed:.1f}s\n")

    if n_valid == 0:
        print("[FAIL] 未采到任何含 player 的有效帧，无法验证。请检查：")
        print("       1) 视频中确有球员；2) 检测引擎(cpp)阈值 0.25 是否漏检远场小球员。")
        det_model.release()
        pose_model.release()
        return

    # ============ 聚合统计 ============
    # 尺度分布
    box_hs = np.array([r['box_h'] for r in records])
    scale_dist = {
        "min": float(box_hs.min()), "median": float(np.median(box_hs)),
        "max": float(box_hs.max()), "n": int(n_valid),
    }
    bins = {}
    for label, fn in SCALE_BINS:
        cnt = int(np.sum([fn(h) for h in box_hs]))
        bins[label] = cnt
    scale_dist["bins"] = bins

    # 对每个 margin M>0：逐帧、逐关键点对比（只统计两边都可见的点）
    report = {"video": args.video, "frames": n_valid, "margins": margins,
              "kpt_conf_thres": kpt_thres, "scale_distribution": scale_dist,
              "per_margin": {}, "by_scale": []}

    def _agg(rows):
        """rows: list of (box_h, res_0, res_M)。返回该子集的聚合 dict。"""
        off = []
        vis_gain = []
        px_loss = []
        for box_h, r0, rm in rows:
            if r0 is None or rm is None:
                continue
            both = r0['visible'] & rm['visible']
            if not np.any(both):
                continue
            d = np.linalg.norm(rm['kpts'][both] - r0['kpts'][both], axis=1)
            off.extend((d / box_h).tolist())               # 归一化偏移（box_h 归一）
            vis_gain.append(int(np.sum(rm['visible']) - int(np.sum(r0['visible']))))
            px_loss.append(1.0 - (box_h * rm['scale']) / (box_h * r0['scale']))
        if not off:
            return None
        return {
            "offset_norm_mean_pct": round(float(np.mean(off)) * 100, 3),
            "offset_norm_p95_pct": round(p95(off) * 100, 3),
            "offset_norm_max_pct": round(float(np.max(off)) * 100, 3),
            "n_kpt_pairs": int(len(off)),
            "human_px_loss_mean_pct": round(float(np.mean(px_loss)) * 100, 3),
            "visible_gain_mean": round(float(np.mean(vis_gain)), 3),
        }

    for m in margins:
        if m == 0:
            continue
        rows = [(r['box_h'], r['per_margin'][0], r['per_margin'][m]) for r in records]
        report["per_margin"][str(m)] = _agg(rows)

    # 按尺度分层
    for label, fn in SCALE_BINS:
        sub = [r for r in records if fn(r['box_h'])]
        if not sub:
            continue
        entry = {"label": label, "n_frames": len(sub), "per_margin": {}}
        for m in margins:
            if m == 0:
                continue
            rows = [(r['box_h'], r['per_margin'][0], r['per_margin'][m]) for r in sub]
            entry["per_margin"][str(m)] = _agg(rows)
        report["by_scale"].append(entry)

    # ============ 判定（以当前值 margin=10 为主，若 10 不在档位则取第一个 M>0）============
    current_margin = 10 if 10 in margins else margins[1]
    cur = report["per_margin"].get(str(current_margin))

    def _verdict(v):
        """单个指标 -> (级别, 文字)。级别 pass/warn/block。"""
        mean = v["offset_norm_mean_pct"]
        p = v["offset_norm_p95_pct"]
        loss = v["human_px_loss_mean_pct"]
        if mean > 2.0 or p > 4.0 or loss > 20.0:
            return "block", "阻塞"
        if mean > 1.0 or p > 2.0 or loss > 10.0:
            return "warn", "关注"
        return "pass", "通过"

    print("=" * 68)
    print("① 人体尺度分布（判断视频是否覆盖远/中/近）")
    print("=" * 68)
    print(f"  box_h: min={scale_dist['min']:.0f}  median={scale_dist['median']:.0f}  "
          f"max={scale_dist['max']:.0f}  (n={n_valid})")
    for label, cnt in bins.items():
        bar = "#" * max(1, int(cnt / max(1, n_valid) * 40))
        print(f"  {label:<24} {cnt:>3} 帧  {bar}")

    print("\n" + "=" * 68)
    print("② 各 margin 档汇总（相对 margin=0 准真值，只比较两边都可见的点）")
    print("=" * 68)
    header = f"  {'margin':<8}{'偏移均值%':>10}{'偏移P95%':>10}{'偏移max%':>10}{'分辨率损失%':>12}{'可见点增量':>12}{'点对数':>8}"
    print(header)
    print("  " + "-" * 66)
    for m in margins:
        if m == 0:
            continue
        v = report["per_margin"].get(str(m))
        if v is None:
            print(f"  {m:<8}{'无共见点':>10}")
            continue
        mark = "  ← 当前值" if m == current_margin else ""
        print(f"  {m:<8}{v['offset_norm_mean_pct']:>10.3f}{v['offset_norm_p95_pct']:>10.3f}"
              f"{v['offset_norm_max_pct']:>10.3f}{v['human_px_loss_mean_pct']:>12.3f}"
              f"{v['visible_gain_mean']:>12.2f}{v['n_kpt_pairs']:>8}{mark}")

    print("\n" + "=" * 68)
    print("③ 按人体尺度分层（远场小框是最坏情况）")
    print("=" * 68)
    for entry in report["by_scale"]:
        print(f"\n  [{entry['label']}]  n={entry['n_frames']} 帧")
        for m in margins:
            if m == 0:
                continue
            v = entry["per_margin"].get(str(m))
            if v is None:
                print(f"    margin={m:<3} 无共见点")
                continue
            print(f"    margin={m:<3} 偏移均值 {v['offset_norm_mean_pct']:.3f}%  "
                  f"P95 {v['offset_norm_p95_pct']:.3f}%  分辨率损失 {v['human_px_loss_mean_pct']:.2f}%")

    print("\n" + "=" * 68)
    print("④ 判定（阈值: 均值<1% / P95<2% / 分辨率损失<10%）")
    print("=" * 68)
    if cur is None:
        print(f"  [FAIL] margin={current_margin} 无有效对比数据")
    else:
        level, txt = _verdict(cur)
        print(f"  当前 margin={current_margin}:")
        print(f"    偏移均值 {cur['offset_norm_mean_pct']:.3f}%  "
              f"(阈值 1.0%)")
        print(f"    偏移 P95  {cur['offset_norm_p95_pct']:.3f}%  "
              f"(阈值 2.0%)")
        print(f"    分辨率损失 {cur['human_px_loss_mean_pct']:.3f}%  "
              f"(阈值 10%)")
        print(f"    可见点增量(收益端) {cur['visible_gain_mean']:+.2f} 点/帧")
        print(f"  => 结论: [{txt.upper()}] "
              + ("margin 扩边副作用可控" if level == "pass" else
                 "margin 扩边副作用偏大，建议评估 margin 自适应或下调 margin"))

    # ============ 落盘报告 ============
    out_dir = args.out or os.path.join(PROJ_ROOT, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"crop_margin_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已落盘: {out_path}")

    det_model.release()
    pose_model.release()


if __name__ == "__main__":
    main()
