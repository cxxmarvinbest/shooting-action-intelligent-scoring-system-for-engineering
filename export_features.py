# -*- coding: utf-8 -*-
"""
特征导出脚本（阶段 2 阈值修正前置）
========================================
逐帧导出「原始检测/姿态特征 + 默认 FSM 状态判别」，供人工标注每帧状态后修正阈值。

导出字段（CSV 宽表 + JSON 结构体，两者逐行一一对应）：
  - 基础：frame_idx（帧号）、ts（视频内相对秒）
  - 状态：fsm_state（用 config/fsm.yaml 默认值跑出的 4 态判别）、real_state（人工回填真实状态）、
          shot_count、shot_event_release_idx（本帧刚触发出手事件时的出手帧号）
  - 框/球：player_box、ball_box（单球最优）、ball_conf
  - 投篮臂侧关节角（与 FSM 判定一致）：shoulder_ang / elbow_ang / hip_ang / knee_ang / ankle_ang
  - 双侧关节角（阶段 2 分析投篮臂侧用）：L_* / R_* 各 4 个角
  - 相对高度：head_ref_y / shoulder_y / hip_y / knee_y / elbow_y / wrist_y / wrist_x / torso_len / player_h
  - 腕-头 Y 偏移：wrist_head_dy_px（像素，正值=手腕在头之上）、wrist_rel_head（归一化 ÷ torso_len）
  - 球腕距离：ball_wrist_dist_px（像素）、ball_wrist_dist_norm（归一化）、
              ball_wrist_intersect（球框四边界是否覆盖腕点）、ball_wrist_relation
  - 其他归一化特征：elbow_rel_shoulder / shoulder_rel_hip / knee_rel_hip
  - 17 关键点：kp00_x/kp00_y/kp00_conf ... kp16_x/kp16_y/kp16_conf（COCO 顺序）

用法（在项目根目录运行，需 RK3588 板端 rknn_toolkit_lite2 环境）：
  python export_features.py data/standard/shot_1.mp4
  python export_features.py data/standard/shot_1.mp4 --out data/output/export_features --stride 1

说明：
  - 依赖 RKNN 检测/姿态模型（VideoAnalyzer.load_models），Windows 本地无 NPU 时无法跑真机，
    仅用于板端导出；列构建逻辑 _build_row 为纯函数，可脱离模型单测。
  - 默认值全部来自 config/fsm.yaml（ShotFSM 无参构造直接读该文件），本脚本不硬编码任何阈值。

依赖：numpy / cv2 / config / common.logger / vision_algorithm.*
"""

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np

# 把项目根目录加入 sys.path，保证可从任意位置运行
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402

from config import Config  # noqa: E402
from common.logger import setup_logger  # noqa: E402
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer  # noqa: E402
from vision_algorithm.pose.pose_feature import calculate_angle  # noqa: E402
from vision_algorithm.segmentation.shot_fsm import ShotFSM  # noqa: E402

logger = logging.getLogger("basketball_scoring")

# COCO 17 关键点索引（与 shot_fsm.py / pose_feature.py 口径一致）
COCO_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]


def _round(v, nd=2):
    """把数值四舍五入到 nd 位小数；None 原样返回（供 CSV 留空 / JSON 存 null）。"""
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _visible_mask(kpts, kpt_conf):
    """关键点可见性掩码：坐标非 (0,0) 且 conf>0（与 pose_feature 口径一致）。"""
    kpts = np.asarray(kpts, dtype=np.float32)
    coord_visible = (kpts[:, 0] > 0) | (kpts[:, 1] > 0)
    if kpt_conf is not None:
        conf = np.asarray(kpt_conf, dtype=np.float32).reshape(-1)
        if len(conf) == kpts.shape[0]:
            return (coord_visible & (conf > 0.0)).tolist()
    return coord_visible.tolist()


def _side_angles(kpts, visible, s, e, w, h, k, a):
    """计算单侧四个关节角（肩/肘/髋/膝）；三点任一不可见 -> 对应角为 None。"""
    out = {}
    if visible[s] and visible[e] and visible[h]:
        out["shoulder"] = _round(calculate_angle(kpts[h], kpts[s], kpts[e]), 2)
    if visible[s] and visible[e] and visible[w]:
        out["elbow"] = _round(calculate_angle(kpts[s], kpts[e], kpts[w]), 2)
    if visible[s] and visible[h] and visible[k]:
        out["hip"] = _round(calculate_angle(kpts[s], kpts[h], kpts[k]), 2)
    if visible[h] and visible[k] and visible[a]:
        out["knee"] = _round(calculate_angle(kpts[h], kpts[k], kpts[a]), 2)
    return out


def _box4(name, box):
    """把 (x1,y1,x2,y2) 拆成 4 个独立列；box 为 None 时全部置 None。"""
    if box is None:
        return {f"{name}_x1": None, f"{name}_y1": None,
                f"{name}_x2": None, f"{name}_y2": None}
    x1, y1, x2, y2 = box
    return {f"{name}_x1": int(round(x1)), f"{name}_y1": int(round(y1)),
            f"{name}_x2": int(round(x2)), f"{name}_y2": int(round(y2))}


def _build_row(fd, fsm_res):
    """由单帧原始特征 fd + FSM 判别结果 fsm_res 组装一行导出数据（纯函数，可单测）。

    参数：
        fd      —— VideoAnalyzer._extract_frame_metrics 返回的当前帧 dict
        fsm_res —— ShotFSM.feed(fd) 返回的 dict（含 state / features / shot_event）
    返回：扁平 dict，所有行的 key 集合一致（缺省为 None），可直接写 CSV/JSON。
    """
    row = {}

    # ── 基础 ──
    row["frame_idx"] = fd.get("idx")
    ts = fd.get("ts")
    row["ts"] = _round(ts, 3)

    # ── FSM 默认状态判别 ──
    row["fsm_state"] = fsm_res.get("state")
    # 人工标注列：紧跟 fsm_state，导出时留空，供用户回填每帧真实状态
    # （IDLE/HOLD/SQUAT_RAISE/OVERHEAD_RELEASE/FOLLOW）
    row["real_state"] = ""
    row["shot_count"] = fsm_res.get("shot_count")
    evt = fsm_res.get("shot_event")
    row["shot_event_release_idx"] = evt.get("release_idx") if evt else None

    # ── 框 / 球（原始检测，未过 FSM 阈值）──
    row.update(_box4("player_box", fd.get("player_box")))
    ball_boxes = fd.get("ball_boxes") or []
    row.update(_box4("ball_box", ball_boxes[0] if ball_boxes else None))
    ball_confs = fd.get("ball_confs") or []
    row["ball_conf"] = _round(ball_confs[0], 4) if ball_confs else None

    # ── FSM 推导特征（投篮臂侧 + 归一化值）──
    f = fsm_res.get("features") or {}
    row["side"] = f.get("side")
    row["torso_len"] = _round(f.get("torso_len"), 2)
    row["player_h"] = _round(fd.get("player_h"), 2)
    row["head_ref_y"] = _round(f.get("head_ref_y"), 2)
    row["shoulder_y"] = _round(f.get("shoulder_y"), 2)
    row["hip_y"] = _round(f.get("hip_y"), 2)
    row["knee_y"] = _round(f.get("knee_y"), 2)
    row["elbow_y"] = _round(f.get("elbow_y"), 2)
    row["wrist_y"] = _round(f.get("wrist_y"), 2)
    row["wrist_x"] = _round(fd.get("wrist_x"), 2)

    # 关节角（投篮臂侧，与 FSM 判定一致）
    row["shoulder_ang"] = _round(f.get("shoulder_ang"), 2)
    row["elbow_ang"] = _round(f.get("elbow_ang"), 2)
    row["hip_ang"] = _round(f.get("hip_ang"), 2)
    row["knee_ang"] = _round(f.get("knee_ang"), 2)
    row["ankle_ang"] = _round(fd.get("ankle_angle"), 2)

    # 腕-头 Y 偏移（像素 + 归一化；正值=手腕在头之上）
    head_ref = f.get("head_ref_y")
    wrist_y = f.get("wrist_y")
    if head_ref is not None and wrist_y is not None:
        row["wrist_head_dy_px"] = _round(float(head_ref) - float(wrist_y), 2)
    else:
        row["wrist_head_dy_px"] = None
    row["wrist_rel_head"] = _round(f.get("wrist_rel_head"), 4)

    # 球腕距离（像素 + 归一化 + 四边界覆盖 + 关系）
    dnorm = f.get("ball_wrist_dist_norm")
    torso = f.get("torso_len")
    row["ball_wrist_dist_px"] = (_round(float(dnorm) * float(torso), 2)
                                 if dnorm is not None and torso else None)
    row["ball_wrist_dist_norm"] = _round(dnorm, 4)
    bi = f.get("ball_wrist_intersect")
    row["ball_wrist_intersect"] = None if bi is None else (1 if bi else 0)
    row["ball_wrist_relation"] = f.get("ball_wrist_relation")

    # 其他归一化相对高度
    row["elbow_rel_shoulder"] = _round(f.get("elbow_rel_shoulder"), 4)
    row["shoulder_rel_hip"] = _round(f.get("shoulder_rel_hip"), 4)
    row["knee_rel_hip"] = _round(f.get("knee_rel_hip"), 4)

    # ── 双侧关节角（阶段 2 分析投篮臂侧用）──
    kpts = fd.get("kpts")
    kpt_conf = fd.get("kpt_conf")
    bilat = ["L_shoulder_ang", "L_elbow_ang", "L_hip_ang", "L_knee_ang",
             "R_shoulder_ang", "R_elbow_ang", "R_hip_ang", "R_knee_ang"]
    if kpts is not None:
        kpts = np.asarray(kpts, dtype=np.float32)
        if kpts.ndim == 2 and kpts.shape[0] >= 17:
            vis = _visible_mask(kpts, kpt_conf)
            L = _side_angles(kpts, vis, 5, 7, 9, 11, 13, 15)
            R = _side_angles(kpts, vis, 6, 8, 10, 12, 14, 16)
            row["L_shoulder_ang"], row["L_elbow_ang"] = L.get("shoulder"), L.get("elbow")
            row["L_hip_ang"], row["L_knee_ang"] = L.get("hip"), L.get("knee")
            row["R_shoulder_ang"], row["R_elbow_ang"] = R.get("shoulder"), R.get("elbow")
            row["R_hip_ang"], row["R_knee_ang"] = R.get("hip"), R.get("knee")
            n_vis = int(sum(vis))
        else:
            for k in bilat:
                row[k] = None
            n_vis = 0
    else:
        for k in bilat:
            row[k] = None
        n_vis = 0
    row["n_visible_kpts"] = n_vis

    # ── 17 关键点（COCO 顺序，坐标 + 置信度）──
    for i in range(17):
        if kpts is not None and kpts.ndim == 2 and kpts.shape[0] > i:
            x, y = float(kpts[i, 0]), float(kpts[i, 1])
            row[f"kp{i:02d}_x"] = int(round(x)) if (x > 0 or y > 0) else None
            row[f"kp{i:02d}_y"] = int(round(y)) if (x > 0 or y > 0) else None
            c = None
            if kpt_conf is not None and len(kpt_conf) > i:
                c = float(kpt_conf[i])
            row[f"kp{i:02d}_conf"] = _round(c, 3)
        else:
            row[f"kp{i:02d}_x"] = None
            row[f"kp{i:02d}_y"] = None
            row[f"kp{i:02d}_conf"] = None

    return row


def _shot_summary(evt):
    """把 shot_event 转成不含 frame_metrics 的 JSON 安全摘要。

    frame_metrics 是内部逐帧 fd 列表（含 kpts 等 numpy 对象），供下游评分引擎
    使用，导出脚本不需要（逐帧数据已在 frames 中完整保存），剥离后可避免
    json.dump 遇 ndarray 报 TypeError。
    """
    return {
        'shot_idx': evt.get('shot_idx'),
        'start_idx': evt.get('start_idx'),
        'release_idx': evt.get('release_idx'),
        'start_time': _round(evt.get('start_time'), 3),
        'end_time': _round(evt.get('end_time'), 3),
        'duration': _round(evt.get('duration'), 3),
        'has_squat': evt.get('has_squat'),
        'state_entries': evt.get('state_entries'),
        'hold_idx': evt.get('hold_idx'),
        'crouch_min_idx': evt.get('crouch_min_idx'),
        'overhead_idx': evt.get('overhead_idx'),
    }


def _json_default(o):
    """JSON 序列化兜底：numpy 数组/标量转 Python 原生类型。"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.str_):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _write_outputs(rows, shots, fsm_cfg, video_path, fps, stride, out_dir):
    """写 CSV（utf-8-sig，Excel 友好）与 JSON（含 meta + frames）。"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(out_dir, f"features_{base}.csv")
    json_path = os.path.join(out_dir, f"features_{base}.json")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    payload = {
        "meta": {
            "video": video_path,
            "fps": _round(fps, 3),
            "stride": stride,
            "frame_count": len(rows),
            "shots": [_shot_summary(s) for s in shots],
            "fsm_config": fsm_cfg,
        },
        "frames": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

    return csv_path, json_path


def run_export(video_path, out_dir, stride=1, max_frames=0):
    """主流程：加载模型 -> 逐帧提取特征 + FSM 判别 -> 写 CSV/JSON。"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    logger.info("=" * 60)
    logger.info("特征导出开始（阶段 2 阈值修正前置）")
    logger.info("视频: %s", video_path)
    logger.info("采样步长=%d, 最多处理=%d 帧(0=全部)", stride, max_frames)

    logger.info("加载 RKNN 检测/姿态模型 ...")
    analyzer = VideoAnalyzer()
    analyzer.load_models()
    analyzer.reset_trackers()
    logger.info("模型加载完成")

    # FSM 默认参数：无参构造直接读 config/fsm.yaml（阶段 1 通用默认值）
    fsm = ShotFSM()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        analyzer.release_models()
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info("视频帧率=%.1f, 总帧≈%d", fps, total_frames)

    rows = []
    shots = []
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        if max_frames > 0 and processed >= max_frames:
            break

        fd = analyzer._extract_frame_metrics(frame, frame_idx, ts=frame_idx / fps)
        fsm_res = fsm.feed(fd)
        rows.append(_build_row(fd, fsm_res))
        if fsm_res.get("shot_event"):
            shots.append(fsm_res["shot_event"])

        if processed % 30 == 0:
            logger.info("已处理 %d 帧 | 当前状态=%s | 累计投篮=%d",
                        processed, fsm_res.get("state"), fsm_res.get("shot_count"))
        processed += 1
        frame_idx += 1

    cap.release()
    analyzer.release_models()

    csv_path, json_path = _write_outputs(
        rows, shots, fsm.cfg, video_path, fps, stride, out_dir)

    logger.info("=" * 60)
    logger.info("特征导出完成: 共 %d 帧", len(rows))
    logger.info("  CSV : %s", csv_path)
    logger.info("  JSON: %s", json_path)
    logger.info("  默认 FSM 检出投篮事件: %d 次", len(shots))
    for evt in shots:
        logger.info("    第 %d 次: 出手帧=%s, 起点帧=%s, 下蹲=%s",
                    evt.get("shot_idx"), evt.get("release_idx"),
                    evt.get("start_idx"), evt.get("has_squat"))

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="投篮特征逐帧导出（FSM 阈值修正前置）")
    parser.add_argument("video", help="标准视频路径（如 data/standard/shot_1.mp4）")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认 data/output/export_features）")
    parser.add_argument("--stride", type=int, default=None,
                        help="采样步长（默认取 Config.FRAME_STRIDE）")
    parser.add_argument("--frames", type=int, default=0,
                        help="最多处理帧数（0=整段视频）")
    parser.add_argument("--log-dir", default=None, help="日志目录（默认 logs）")
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(PROJECT_ROOT, "logs")
    setup_logger(log_dir)

    out_dir = args.out or os.path.join(Config.OUTPUT_DIR, "export_features")
    stride = max(1, int(args.stride) if args.stride is not None
                 else int(Config.FRAME_STRIDE))

    try:
        run_export(args.video, out_dir, stride=stride, max_frames=args.frames)
        logger.info("=" * 60)
        logger.info("特征导出流程结束")
    except Exception:
        logger.exception("特征导出失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
