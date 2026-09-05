# -*- coding: utf-8 -*-
"""
T2 Anti-flicker 跟踪层 A/B 回归：tracker ON（启用） vs OFF（回退纯面积选择）
========================================================================================
回归本质 = A/B 对比：ANTI_FLICKER_ENABLE=true（改动后） vs false（改动前基线），
唯一变量是「主球员框选择逻辑是否经过 SingleTargetTracker 时序稳定」。

要验证的四个目标（对应 T2 设计）：
  1. 无 flicker      —— 关键点帧间位移 P95 + 可见性闪烁(1→0→1)次数，ON 应 ≤ OFF
  2. 误检抑制         —— 主球员框帧间 IoU 跳变次数（身份跳变），ON 应 < OFF
  3. 漏检兜底         —— player 命中帧数 + 可见点总数 + 断链次数，ON 应 ≥/≤ OFF
  4. 检测不被污染     —— ball 检测数（tracker 不碰 dets，只改「选哪个 player 框」）
  （附加）评分稳定    —— idx_squat（动作分段）一致（--full 跑端到端）

两条链路都测（复用真实链路，同 T2-5 margin 回归）：
  离线：_extract_frame_metrics(frame)                     # 原图 → 640canvas → 反算原图 → crop → 320pose
  实时：_extract_frame_metrics_norm(det360, preview)      # cv2.resize 模拟 RGA（det360=640x360, preview=1280x720）

关键实现细节：
  tracker 开关用实例属性 analyzer._tracker_enable（True/False），不用改 Config；
  每次切换开关后必须 analyzer.reset_trackers() 清空锁定状态，保证两遍采样同起点。

用法（RK3588 板端，需 rknn-toolkit-lite2 与 rknn_yolov8 .so 已部署）：
  python test/regression/test_t2_5_antiflicker.py --video test/test_videos/left_rtsp.mp4 --frames 50   # 层面 A
  python test/regression/test_t2_5_antiflicker.py --video test/test_videos/left_rtsp.mp4 --full        # 层面 A+B

判定阈值（T2 Anti-flicker 已锁定）：
  flicker-位移   ON P95 ≤ OFF P95×1.1（>1.1× 说明跟踪器反而引入抖动，WARN/BLOCK）
  误检突变       ON 框跳变次数 ≤ OFF（> WARN；跟踪器应抑制身份跳变）
  漏检兜底       ON player命中 ≥ OFF、ON 可见点 ≥ OFF（< 则 WARN；兜底应减少整帧丢失）
  断链次数       ON ≤ OFF（> WARN）
  检测一致性     ball 检测数差异（观察项，受 player_box 约束 + RKNN 非确定性影响，不硬判）
  评分稳定       idx_squat ±2 帧内（--full）
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
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import Config
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer

DEFAULT_VIDEO = os.path.join(PROJ_ROOT, "test", "left_side_basketball.mp4")
NUM_KPTS = 17

CHAIN_DEFS = {
    "off": {"name": "离线(原图)", "res": (1920, 1080)},
    "rt":  {"name": "实时(预览)", "res": (1280, 720)},
}


def p95(a):
    """第 95 百分位（空数组返回 nan）。"""
    return float(np.percentile(a, 95)) if len(a) else float("nan")


def box_iou(a, b):
    """两框 IoU（任一 None 返回 None）。"""
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
    """可见性闪烁次数：可见→不可见→可见（1→0→1 单帧闪断），按关键点独立统计。"""
    cnt = 0
    for k in range(NUM_KPTS):
        for i in range(1, len(vis_seq) - 1):
            if vis_seq[i - 1][k] and (not vis_seq[i][k]) and vis_seq[i + 1][k]:
                cnt += 1
    return cnt


def _interframe_disp(fm):
    """相邻采样帧关键点位移（÷当前帧 box_h 归一化），衡量空间抖动。"""
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


def _count_box_jumps(fm, iou_thresh=0.3):
    """相邻帧主球员框 IoU < thresh 的跳变次数（身份跳变/误检抢身份近似）。"""
    cnt = 0
    for i in range(1, len(fm)):
        a, b = fm[i - 1]['player_box'], fm[i]['player_box']
        if a is None or b is None:
            continue
        if box_iou(a, b) < iou_thresh:
            cnt += 1
    return cnt


def _count_dropouts(fm):
    """player_box 从有到 None 的断链次数（漏检导致整帧丢主球员）。"""
    cnt = 0
    for i in range(1, len(fm)):
        if fm[i - 1]['player_box'] is not None and fm[i]['player_box'] is None:
            cnt += 1
    return cnt


def run_level_a(analyzer, args):
    """层面 A：逐帧回归（离线 + 实时两链路，tracker ON/OFF 各跑一遍）。"""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FAIL] 打开视频失败: {args.video}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, total // max(1, args.frames))
    print(f"视频信息   : total≈{total} 帧, fps={fps:.1f}, 采样步长={stride}")

    results = {"off": {'off': [], 'rt': []}, "on": {'off': [], 'rt': []}}

    for mode in ("off", "on"):
        # 切换开关 + 清空跟踪器状态，保证两遍采样从「无目标」同起点开始
        analyzer._tracker_enable = (mode == "on")
        analyzer.reset_trackers()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx, sampled = 0, 0
        t_start = time.perf_counter()
        while sampled < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            # 模拟 RGA 两路输出（离线原图 + 实时 640x360/1280x720）
            preview = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            det360 = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            ts = frame_idx / fps

            fd_off = analyzer._extract_frame_metrics(frame, frame_idx, ts=ts)
            fd_rt = analyzer._extract_frame_metrics_norm(det360, preview, frame_idx, ts=ts)
            results[mode]['off'].append(_pack_frame(fd_off))
            results[mode]['rt'].append(_pack_frame(fd_rt))

            sampled += 1
            frame_idx += 1
            if sampled % 10 == 0:
                print(f"  [tracker {'ON ' if mode == 'on' else 'OFF'}] 已采 {sampled}/{args.frames} 帧")

        print(f"  [tracker {'ON ' if mode == 'on' else 'OFF'}] 采样完成 {sampled} 帧，"
              f"耗时 {time.perf_counter() - t_start:.1f}s")

    cap.release()
    # 恢复默认开关（配置值）
    analyzer._tracker_enable = bool(Config.get("ANTI_FLICKER_ENABLE", True))
    analyzer.reset_trackers()
    return results


def _analyze_chain(fm_off, fm_on):
    """对单条链路（off/rt）对比 tracker OFF vs ON。返回聚合 dict。"""
    n = len(fm_off)

    def _summ(fm):
        vis_total = sum(int(f['vis'].sum()) for f in fm if f['vis'] is not None)
        player_hits = sum(1 for f in fm if f['player_box'] is not None)
        ball_total = sum(f['ball_count'] for f in fm)
        kpts_hits = sum(1 for f in fm if f['kpts'] is not None)
        vis_seq = [f['vis'] for f in fm if f['vis'] is not None]
        disp = _interframe_disp(fm)
        return {
            "vis_total": vis_total, "player_hits": player_hits,
            "ball_total": ball_total, "kpts_hits": kpts_hits,
            "flicker_count": _count_flicker(vis_seq),
            "box_jumps": _count_box_jumps(fm),
            "dropouts": _count_dropouts(fm),
            "disp_mean_pct": round(float(np.mean(disp)) * 100, 3) if disp else None,
            "disp_p95_pct": round(p95(disp) * 100, 3) if disp else None,
            "disp_n": len(disp),
        }

    s_off, s_on = _summ(fm_off), _summ(fm_on)

    # ball 检测一致性（tracker 不碰 dets，但 ball 过滤受 player_box 约束，观察项）
    ball_mismatch = sum(1 for i in range(n)
                        if fm_off[i]['ball_count'] != fm_on[i]['ball_count'])

    return {
        "n_frames": n,
        "off": s_off, "on": s_on,
        "ball_mismatch": ball_mismatch,
    }


def _verdict(s_off, s_on):
    """逐指标判定，返回 list of (label, level, text)。"""
    rows = []

    # ① flicker 帧间位移
    d0, d1 = s_off["disp_p95_pct"], s_on["disp_p95_pct"]
    if d0 is not None and d1 is not None:
        ratio = d1 / max(1e-9, d0)
        lvl = "PASS" if ratio <= 1.1 else ("WARN" if ratio <= 1.3 else "BLOCK")
        rows.append(("flicker位移", lvl, f"ON/OFF P95 比值 {ratio:.2f} (阈值 1.1)"))

    # ② 误检突变（身份跳变）
    j0, j1 = s_off["box_jumps"], s_on["box_jumps"]
    lvl = "PASS" if j1 <= j0 else "WARN"
    rows.append(("误检突变", lvl, f"框跳变次数 ON={j1} vs OFF={j0}"))

    # ③ 漏检兜底（player 命中 + 可见点）
    ph0, ph1 = s_off["player_hits"], s_on["player_hits"]
    v0, v1 = s_off["vis_total"], s_on["vis_total"]
    lvl = "PASS" if (ph1 >= ph0 and v1 >= v0) else "WARN"
    rows.append(("漏检兜底", lvl,
                 f"player命中 ON={ph1} vs OFF={ph0}；可见点 ON={v1} vs OFF={v0}"))

    # ④ 断链次数
    dp0, dp1 = s_off["dropouts"], s_on["dropouts"]
    lvl = "PASS" if dp1 <= dp0 else "WARN"
    rows.append(("断链次数", lvl, f"ON={dp1} vs OFF={dp0}"))

    # ⑤ flicker 可见性闪烁
    f0, f1 = s_off["flicker_count"], s_on["flicker_count"]
    lvl = "PASS" if f1 <= f0 else "WARN"
    rows.append(("可见性闪烁", lvl, f"ON={f1} vs OFF={f0}"))

    return rows


def run_level_b(analyzer, args):
    """层面 B：端到端评分回归（完整视频 process_video 跑两遍），对比 idx_squat。"""
    print("\n" + "=" * 68)
    print("层面 B：端到端评分回归（完整视频 process_video，save_visuals=False）")
    print("=" * 68)
    out = {}
    for mode in ("off", "on"):
        analyzer._tracker_enable = (mode == "on")
        analyzer.reset_trackers()
        t0 = time.perf_counter()
        _, _, _, _, rel_height, frame_metrics = analyzer.process_video(
            args.video, save_visuals=False)
        dt = time.perf_counter() - t0
        _, _, _, idx_squat = VideoAnalyzer._split_and_height(frame_metrics)
        out[mode] = {'idx_squat': idx_squat, 'rel_height': rel_height,
                     'n_frames': len(frame_metrics), 'elapsed': dt}
        print(f"  tracker {'ON ' if mode == 'on' else 'OFF'} idx_squat={idx_squat:<5} "
              f"rel_height={rel_height:.4f}  帧数={len(frame_metrics):<5} 耗时 {dt:.1f}s")
    analyzer._tracker_enable = bool(Config.get("ANTI_FLICKER_ENABLE", True))
    analyzer.reset_trackers()
    return out


def main():
    ap = argparse.ArgumentParser(description="T2 Anti-flicker 跟踪层 A/B 回归（tracker ON/OFF）")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="测试视频路径")
    ap.add_argument("--frames", type=int, default=50, help="层面 A 采样帧数（默认 50）")
    ap.add_argument("--full", action="store_true", help="额外跑层面 B 端到端评分回归")
    ap.add_argument("--out", default=None, help="报告落盘目录（默认 data/output）")
    args = ap.parse_args()

    print("=" * 68)
    print("T2 Anti-flicker 跟踪层 A/B 回归（tracker ON vs OFF）")
    print("=" * 68)
    print(f"视频       : {args.video}")
    print(f"采样帧数   : {args.frames}")
    print(f"检测引擎   : {Config.get('DET_ENGINE', 'cpp')}")
    print(f"姿态模型   : {Config.POSE_RKNN_PATH}  (input {Config.POSE_MODEL_W}x{Config.POSE_MODEL_H})")
    print(f"跟踪参数   : iou={Config.get('TRACKER_IOU_THRESH')} "
          f"hangover={Config.get('TRACKER_HANGOVER')} "
          f"confirm={Config.get('TRACKER_CONFIRM')} "
          f"ema={Config.get('TRACKER_EMA_ALPHA')}\n")

    analyzer = VideoAnalyzer()
    analyzer.load_models()

    report = {"video": args.video, "frames": args.frames,
              "tracker_params": {
                  "iou_thresh": Config.get("TRACKER_IOU_THRESH"),
                  "hangover": Config.get("TRACKER_HANGOVER"),
                  "confirm": Config.get("TRACKER_CONFIRM"),
                  "ema_alpha": Config.get("TRACKER_EMA_ALPHA"),
              }}

    # ================= 层面 A =================
    print("=" * 68)
    print("层面 A：逐帧回归（离线 + 实时两链路，tracker ON/OFF 各一遍）")
    print("=" * 68)
    results = run_level_a(analyzer, args)
    if results is None:
        analyzer.release_models()
        return

    report["level_a"] = {}
    for chain_key in ("off", "rt"):
        chain = _analyze_chain(results["off"][chain_key], results["on"][chain_key])
        report["level_a"][chain_key] = chain

        name = CHAIN_DEFS[chain_key]["name"]
        print(f"\n--- {name}链路 ---  (n={chain['n_frames']} 帧)")
        s_off, s_on = chain["off"], chain["on"]

        print(f"  ① flicker-帧间位移（相邻采样帧关键点位移，÷box_h）")
        print(f"    OFF  均值 {s_off['disp_mean_pct']}%  P95 {s_off['disp_p95_pct']}%  "
              f"(样本 {s_off['disp_n']})")
        print(f"    ON   均值 {s_on['disp_mean_pct']}%  P95 {s_on['disp_p95_pct']}%  "
              f"(样本 {s_on['disp_n']})")

        print(f"  ② 误检突变（相邻帧主球员框 IoU<0.3 的跳变次数，身份跳变近似）")
        print(f"    OFF={s_off['box_jumps']}   ON={s_on['box_jumps']}")

        print(f"  ③ 漏检兜底（player 命中 / 可见点 / 断链）")
        print(f"    OFF  player命中={s_off['player_hits']}  可见点={s_off['vis_total']}  "
              f"断链={s_off['dropouts']}")
        print(f"    ON   player命中={s_on['player_hits']}  可见点={s_on['vis_total']}  "
              f"断链={s_on['dropouts']}")

        print(f"  ④ 可见性闪烁（1→0→1 单帧闪断）")
        print(f"    OFF={s_off['flicker_count']}   ON={s_on['flicker_count']}")

        print(f"  ⑤ 检测不被污染（ball 检测数差异，观察项）")
        print(f"    ball_mismatch={chain['ball_mismatch']} 帧  "
              f"(OFF ball总数={s_off['ball_total']}, ON ball总数={s_on['ball_total']})")

        print(f"  ⑥ 判定")
        for label, lvl, txt in _verdict(s_off, s_on):
            print(f"    [{lvl:<5}] {label:<10} {txt}")

    # ================= 层面 B =================
    if args.full:
        report["level_b"] = run_level_b(analyzer, args)
        lb = report["level_b"]
        d = abs(lb["on"]["idx_squat"] - lb["off"]["idx_squat"])
        ok = d <= 2
        print(f"\n  ⑦ 端到端评分稳定性（idx_squat）")
        print(f"    OFF={lb['off']['idx_squat']}  ON={lb['on']['idx_squat']}  (差 {d})")
        print(f"    => {'PASS' if ok else 'BLOCK'}  (阈值 ±2 帧)")

    # ================= 落盘报告 =================
    out_dir = args.out or os.path.join(PROJ_ROOT, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"t2_antiflicker_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已落盘: {out_path}")

    analyzer.release_models()


if __name__ == "__main__":
    main()
