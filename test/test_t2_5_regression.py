# -*- coding: utf-8 -*-
"""
T2-5 端到端回归：margin 扩边 0→10 全链路无 flicker / 关键点偏移 / 漏检回退 + 评分稳定
========================================================================================
回归本质 = A/B 对比：margin=0（改动前基线） vs margin=10（当前生效值），唯一变量是 margin。

覆盖三条验证维度（对应 T2 链收尾目标）：
  1. 无 flicker      —— 时序：关键点帧间位移 P95 + 可见性闪烁(1→0→1)次数
  2. 关键点偏移       —— 单帧：offset_norm = ‖kpts_10 − kpts_0‖₂ / box_h（只比两边可见点）
  3. 漏检回退         —— 检测一致性(player/ball 检出必须 100% 一致) + 关键点可见点数
  （附加）评分稳定    —— 端到端：idx_squat / rel_height / 四关节角度曲线 RMSE

两条链路都测（复用真实链路）：
  离线：_extract_frame_metrics(frame)            # 原图1920x1080 → 640canvas → 反算原图 → crop → 320pose
  实时：_extract_frame_metrics_norm(det360, preview)  # cv2.resize 模拟 RGA（det360=640x360, preview=1280x720）

关键实现细节：
  Config 是只读对象（__setattr__ 抛异常），且 POSE_CROP_MARGIN 不在 ENV_OVERRIDES 环境变量表，
  两条链路内部都是 int(Config.get("POSE_CROP_MARGIN", 10))，故切 margin 只能改 Config._data。

用法（RK3588 板端，需 rknn-toolkit-lite2 与 rknn_yolov8 .so 已部署）：
  python test/test_t2_5_regression.py --video test/test_videos/left_rtsp.mp4 --frames 50   # 层面 A
  python test/test_t2_5_regression.py --video test/test_videos/left_rtsp.mp4 --full        # 层面 A+B

判定阈值（T2-5 已锁定）：
  检测一致性   player/ball 检出 100% 相等（任何不一致=BLOCK，说明 Config 污染了检测）
  可见点计数   m10 ≥ m0（回退=BLOCK，与 T2-3 收益预期矛盾）
  关键点偏移   均值<1% 且 P95<2%（1~2% WARN，>2% BLOCK）
  flicker-位移 m10 P95 ≤ m0 P95×1.1（1.1~1.3× WARN，>1.3× BLOCK）
  flicker-翻转 m10 ≤ m0（> WARN）
  评分稳定     idx_squat±1帧、rel_height<1%、角度RMSE<1°（超限 BLOCK）
========================================================================================
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# 项目根目录加入 sys.path（脚本位于 test/，需能 import config 与 vision_algorithm）
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import Config
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer

DEFAULT_VIDEO = os.path.join(PROJ_ROOT, "test", "left_side_basketball.mp4")
NUM_KPTS = 17
BASELINE_MARGIN = 0
CURRENT_MARGIN = int(Config.get("POSE_CROP_MARGIN", 10))   # 当前生效值，跑完恢复

# 两条链路各自的预览基准（坐标归一化用各自的 box_h，跨链路可比）
CHAIN_DEFS = {
    "off": {"name": "离线(原图)",   "res": (1920, 1080)},
    "rt":  {"name": "实时(预览)",   "res": (1280, 720)},
}


def p95(a):
    """第 95 百分位（空数组返回 nan）。"""
    return float(np.percentile(a, 95)) if len(a) else float("nan")


def set_margin(m):
    """切换 POSE_CROP_MARGIN（Config 只读，直接改私有 _data；两条链路内 Config.get 生效）。"""
    Config._data["POSE_CROP_MARGIN"] = int(m)


def box_iou(a, b):
    """两框 IoU（a/b 均可为 None；任一 None 返回 None）。"""
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pack_frame(fd):
    """从 _extract_frame_metrics / _norm 返回的 frame_metrics 提取回归所需字段。"""
    kpts = fd.get('kpts')
    player_box = fd.get('player_box')
    ball_count = len(fd.get('ball_boxes') or [])
    box_h = float(player_box[3] - player_box[1]) if player_box is not None else None
    vis = None
    if kpts is not None:
        vis = ((kpts[:, 0] > 0) | (kpts[:, 1] > 0))
    return {'idx': fd.get('idx'), 'box_h': box_h, 'player_box': player_box,
            'ball_count': ball_count, 'kpts': kpts, 'vis': vis}


def _count_flicker(vis_seq):
    """可见性闪烁次数：可见→不可见→可见（1→0→1 单帧闪断），按关键点独立统计。

    vis_seq: list of (17,) bool 数组（按时间序）。只统计「前后可见、中间单帧不可见」，
             这是最纯粹的 flicker（时有时无）；连续多帧丢失不算（那是遮挡/丢失）。
    """
    cnt = 0
    for k in range(NUM_KPTS):
        for i in range(1, len(vis_seq) - 1):
            if vis_seq[i - 1][k] and (not vis_seq[i][k]) and vis_seq[i + 1][k]:
                cnt += 1
    return cnt


def _interframe_disp(fm):
    """相邻采样帧关键点位移（÷当前帧 box_h 归一化），衡量空间抖动。

    fm: 逐帧 list。只统计相邻两帧都可见的关键点。
    """
    disp = []
    for i in range(1, len(fm)):
        prev, cur = fm[i - 1], fm[i]
        if prev['kpts'] is None or cur['kpts'] is None:
            continue
        if cur['box_h'] is None or cur['box_h'] <= 0:
            continue
        both = prev['vis'] & cur['vis']
        if not np.any(both):
            continue
        d = np.linalg.norm(cur['kpts'][both] - prev['kpts'][both], axis=1)
        disp.extend((d / cur['box_h']).tolist())
    return disp


def _analyze_chain(frames_by_margin, margins, chain_key):
    """对单条链路（off/rt）做完整回归分析。

    frames_by_margin[m] = 该 margin 的逐帧 list（同一采样序列，按 index 逐帧对齐）。
    """
    base = frames_by_margin[0]   # margin=0 基线
    n = len(base)
    out = {"chain": chain_key, "name": CHAIN_DEFS[chain_key]["name"],
           "n_frames": n, "per_margin": {}, "det_consistency": {}, "offset": {}}

    # ---- 每个 margin 自身的可见点 / flicker ----
    for m in margins:
        fm = frames_by_margin[m]
        vis_total = sum(int(f['vis'].sum()) for f in fm if f['vis'] is not None)
        player_hits = sum(1 for f in fm if f['player_box'] is not None)
        ball_total = sum(f['ball_count'] for f in fm)
        kpts_hits = sum(1 for f in fm if f['kpts'] is not None)
        vis_seq = [f['vis'] for f in fm if f['vis'] is not None]
        disp = _interframe_disp(fm)
        out["per_margin"][str(m)] = {
            "vis_total": vis_total, "player_hits": player_hits,
            "ball_total": ball_total, "kpts_hits": kpts_hits,
            "flicker_count": _count_flicker(vis_seq),
            "disp_mean_pct": round(float(np.mean(disp)) * 100, 3) if disp else None,
            "disp_p95_pct": round(p95(disp) * 100, 3) if disp else None,
            "disp_n": len(disp),
        }

    # ---- 检测一致性（margin=0 vs 每个 m>0，margin 不应影响检测）----
    for m in margins:
        if m == 0:
            continue
        fm = frames_by_margin[m]
        ious, player_mismatch, ball_mismatch = [], 0, 0
        for i in range(n):
            b0, bm = base[i], fm[i]
            p0, pm = b0['player_box'], bm['player_box']
            if (p0 is None) != (pm is None):
                player_mismatch += 1
            elif p0 is not None:
                ious.append(box_iou(p0, pm))
            if b0['ball_count'] != bm['ball_count']:
                ball_mismatch += 1
        out["det_consistency"][str(m)] = {
            "player_mismatch": player_mismatch, "ball_mismatch": ball_mismatch,
            "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "iou_min": round(float(np.min(ious)), 4) if ious else None,
        }

    # ---- 关键点偏移（margin=0 vs m>0，只比两边可见点，÷box_h 归一化）----
    for m in margins:
        if m == 0:
            continue
        fm = frames_by_margin[m]
        offs = []
        for i in range(n):
            b0, bm = base[i], fm[i]
            if b0['kpts'] is None or bm['kpts'] is None:
                continue
            if b0['box_h'] is None or b0['box_h'] <= 0:
                continue
            both = b0['vis'] & bm['vis']
            if not np.any(both):
                continue
            d = np.linalg.norm(bm['kpts'][both] - b0['kpts'][both], axis=1)
            offs.extend((d / b0['box_h']).tolist())
        out["offset"][str(m)] = {
            "mean_pct": round(float(np.mean(offs)) * 100, 3) if offs else None,
            "p95_pct": round(p95(offs) * 100, 3) if offs else None,
            "max_pct": round(float(np.max(offs)) * 100, 3) if offs else None,
            "n": len(offs),
        }
    return out


def run_level_a(analyzer, args, margins):
    """层面 A：逐帧回归（离线 + 实时两链路），返回 {margin: {'off': [...], 'rt': [...]}}。"""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FAIL] 打开视频失败: {args.video}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, total // max(1, args.frames))
    print(f"视频信息   : total≈{total} 帧, fps={fps:.1f}, 采样步长={stride}")

    results = {m: {'off': [], 'rt': []} for m in margins}

    frame_idx, sampled = 0, 0
    t_start = time.perf_counter()
    while sampled < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # 模拟 RGA 两路输出（循环外算一次，不依赖 margin）
        preview = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        det360 = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        ts = frame_idx / fps

        for m in margins:
            set_margin(m)
            fd_off = analyzer._extract_frame_metrics(frame, frame_idx, ts=ts)
            fd_rt = analyzer._extract_frame_metrics_norm(det360, preview, frame_idx, ts=ts)
            results[m]['off'].append(_pack_frame(fd_off))
            results[m]['rt'].append(_pack_frame(fd_rt))

        sampled += 1
        frame_idx += 1
        if sampled % 10 == 0:
            print(f"  已采 {sampled}/{args.frames} 帧")

    cap.release()
    set_margin(CURRENT_MARGIN)   # 恢复默认
    print(f"层面 A 采样完成: {sampled} 帧，耗时 {time.perf_counter() - t_start:.1f}s")
    return results


def _angles_rmse(seq0, seqm):
    """两 margin 的角度序列逐帧 RMSE（度）。seq 为 list of [shoulder,elbow,hip,knee] 或 None。"""
    errs = []
    for x0, xm in zip(seq0, seqm):
        if x0 is None or xm is None:
            continue
        errs.append(float(np.sqrt(np.mean((np.array(x0) - np.array(xm)) ** 2))))
    return round(float(np.mean(errs)), 3) if errs else None


def run_level_b(analyzer, args, margins):
    """层面 B：端到端评分回归（完整视频 process_video 跑两遍），对比评分输出稳定性。"""
    print("\n" + "=" * 68)
    print("层面 B：端到端评分回归（完整视频 process_video，save_visuals=False）")
    print("=" * 68)
    out = {}
    for m in margins:
        set_margin(m)
        t0 = time.perf_counter()
        _, _, _, _, rel_height, frame_metrics = analyzer.process_video(
            args.video, save_visuals=False)
        dt = time.perf_counter() - t0
        _, _, _, idx_squat = VideoAnalyzer._split_and_height(frame_metrics)
        angles = [fm.get('angles') for fm in frame_metrics]
        out[m] = {'idx_squat': idx_squat, 'rel_height': rel_height,
                  'angles': angles, 'n_frames': len(frame_metrics), 'elapsed': dt}
        print(f"  margin={m:<3} idx_squat={idx_squat:<5} rel_height={rel_height:.4f}  "
              f"帧数={len(frame_metrics):<5} 耗时 {dt:.1f}s")
    set_margin(CURRENT_MARGIN)   # 恢复默认

    # 对比（以 margin=0 为基线）
    base = out.get(BASELINE_MARGIN)
    cmp = {}
    for m in margins:
        if m == 0 or base is None or m not in out:
            continue
        b, mm = base, out[m]
        cmp[str(m)] = {
            "idx_squat_delta": int(mm['idx_squat']) - int(b['idx_squat']),
            "rel_height_delta_pct": round(
                abs(mm['rel_height'] - b['rel_height']) / max(1e-9, abs(b['rel_height'])) * 100, 3),
            "angles_rmse_deg": _angles_rmse(b['angles'], mm['angles']),
        }
    out["_baseline_idx_squat"] = base['idx_squat'] if base else None
    out["_baseline_rel_height"] = base['rel_height'] if base else None
    out["_compare"] = cmp
    return out


def _level(name, cond):
    return "PASS" if cond else name


def main():
    ap = argparse.ArgumentParser(description="T2-5 端到端回归（margin 扩边 0→10）")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="测试视频路径")
    ap.add_argument("--frames", type=int, default=50, help="层面 A 采样帧数（默认 50）")
    ap.add_argument("--margins", default="0,10", help="margin 档位，逗号分隔（默认 0,10）")
    ap.add_argument("--full", action="store_true", help="额外跑层面 B 端到端评分回归")
    ap.add_argument("--out", default=None, help="报告落盘目录（默认 data/output）")
    args = ap.parse_args()

    margins = [int(x) for x in args.margins.split(",") if x.strip() != ""]
    if 0 not in margins:
        margins = [0] + margins
    margins = sorted(set(margins))
    # 判定用的「当前 margin」：优先取配置当前值，若不在档位里则退到第一个非 0 档
    cur_m = CURRENT_MARGIN if CURRENT_MARGIN in margins else margins[1]

    print("=" * 68)
    print("T2-5 端到端回归（margin 扩边 0→10，A/B 对比）")
    print("=" * 68)
    print(f"视频       : {args.video}")
    print(f"采样帧数   : {args.frames}")
    print(f"margin 档位: {margins}  (0=基线, {cur_m}=当前值)")
    print(f"检测引擎   : {Config.get('DET_ENGINE', 'cpp')}")
    print(f"姿态模型   : {Config.POSE_RKNN_PATH}  (input {Config.POSE_MODEL_W}x{Config.POSE_MODEL_H})\n")

    analyzer = VideoAnalyzer()
    analyzer.load_models()

    report = {"video": args.video, "frames": args.frames, "margins": margins,
              "current_margin": CURRENT_MARGIN}

    # ================= 层面 A =================
    print("=" * 68)
    print("层面 A：逐帧回归（离线 + 实时两链路）")
    print("=" * 68)
    results = run_level_a(analyzer, args, margins)
    if results is None:
        analyzer.release_models()
        return

    report["level_a"] = {}
    for chain_key in ("off", "rt"):
        frames_by_margin = {m: results[m][chain_key] for m in margins}
        chain = _analyze_chain(frames_by_margin, margins, chain_key)
        report["level_a"][chain_key] = chain

    # ---- 终端打印层面 A ----
    for chain_key in ("off", "rt"):
        chain = report["level_a"][chain_key]
        name = chain["name"]
        print(f"\n--- {name}链路 ---  (n={chain['n_frames']} 帧)")

        # ① 检测一致性
        print(f"  ① 检测一致性（margin 不应影响检测）")
        for m in margins:
            if m == 0:
                continue
            dc = chain["det_consistency"][str(m)]
            iou_txt = f"{dc['iou_mean']}" if dc['iou_mean'] is not None else "N/A"
            ok = dc["player_mismatch"] == 0 and dc["ball_mismatch"] == 0
            print(f"    m0 vs m{m:<3} player检出差异={dc['player_mismatch']}  "
                  f"ball差异={dc['ball_mismatch']}  player_box IoU均值={iou_txt}  "
                  f"=> {'PASS' if ok else 'BLOCK'}")

        # ② 可见点 / ③ 偏移 / ④⑤ flicker 汇总
        pm0 = chain["per_margin"]["0"]
        print(f"  ② 关键点可见性（漏检回退）")
        for m in margins:
            pm = chain["per_margin"][str(m)]
            print(f"    m{m:<3} 可见点总数={pm['vis_total']:<5} player命中={pm['player_hits']}  "
                  f"ball总数={pm['ball_total']} kpts命中={pm['kpts_hits']}")
        delta = chain["per_margin"][str(cur_m)]["vis_total"] - pm0["vis_total"]
        print(f"    => m{cur_m} vs m0 可见点增量 {delta:+d}  "
              f"{'PASS(无回退)' if delta >= 0 else 'BLOCK(回退)'}")

        print(f"  ③ 关键点偏移（m0 vs m{cur_m}，÷box_h 归一化）")
        off = chain["offset"].get(str(cur_m))
        if off and off["mean_pct"] is not None:
            print(f"    均值 {off['mean_pct']}% / P95 {off['p95_pct']}% / max {off['max_pct']}%  "
                  f"(阈值 1%/2%/5%)  点对数={off['n']}")
        else:
            print(f"    无共见点数据")

        print(f"  ④ flicker-帧间位移（相邻采样帧关键点位移，÷box_h）")
        for m in margins:
            pm = chain["per_margin"][str(m)]
            print(f"    m{m:<3} 均值 {pm['disp_mean_pct']}%  P95 {pm['disp_p95_pct']}%  "
                  f"(样本 {pm['disp_n']})")
        dp0 = pm0["disp_p95_pct"]
        dpc = chain["per_margin"][str(cur_m)]["disp_p95_pct"]
        if dp0 is not None and dpc is not None:
            ratio = dpc / max(1e-9, dp0)
            lvl = "PASS" if ratio <= 1.1 else ("WARN" if ratio <= 1.3 else "BLOCK")
            print(f"    => m{cur_m}/m0 P95 比值 {ratio:.2f}  (阈值 1.1)  {lvl}")

        print(f"  ⑤ flicker-可见性闪烁（1→0→1 单帧闪断次数）")
        for m in margins:
            pm = chain["per_margin"][str(m)]
            print(f"    m{m:<3} 闪烁次数 {pm['flicker_count']}")
        f0 = pm0["flicker_count"]
        fc = chain["per_margin"][str(cur_m)]["flicker_count"]
        print(f"    => m{cur_m} vs m0: {fc} vs {f0}  "
              f"{'PASS' if fc <= f0 else 'WARN(上升)'}")

    # ================= 层面 B =================
    if args.full:
        report["level_b"] = run_level_b(analyzer, args, margins)
        lb = report["level_b"]
        cmp = lb.get("_compare", {}).get(str(cur_m))
        cur_squat = lb.get(cur_m, {}).get("idx_squat")
        print(f"\n  ⑥ 端到端评分稳定性（m0 vs m{cur_m}）")
        print(f"    idx_squat: {lb.get('_baseline_idx_squat')} vs {cur_squat} "
              f"（差 {cmp.get('idx_squat_delta') if cmp else 'N/A'}）")
        if cmp:
            print(f"    rel_height 偏差: {cmp['rel_height_delta_pct']}%  (阈值 1%)")
            print(f"    角度曲线 RMSE : {cmp['angles_rmse_deg']}°  (阈值 1°)")
            ok = (abs(cmp['idx_squat_delta']) <= 1 and cmp['rel_height_delta_pct'] < 1.0
                  and (cmp['angles_rmse_deg'] is None or cmp['angles_rmse_deg'] < 1.0))
            print(f"    => {'PASS' if ok else 'BLOCK'}")

    # ================= 落盘报告 =================
    out_dir = args.out or os.path.join(PROJ_ROOT, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"t2_5_regression_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已落盘: {out_path}")

    analyzer.release_models()


if __name__ == "__main__":
    main()
