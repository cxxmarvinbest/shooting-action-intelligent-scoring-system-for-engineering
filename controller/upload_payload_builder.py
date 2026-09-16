# -*- coding: utf-8 -*-
"""
上传 payload 组装（controller/upload_payload_builder）
======================================================
职责：把项目内部数据结构（session_meta / shot_result / 磁盘 data.json / 视频文件）
      映射为后端 IoT addAlgorithm 接口的请求体。字段重命名集中在这一层，便于单测。

映射口径（与后端接口约定一致）：
  - final_data（会话总得分，取各投篮均值）：
      final_score / stage1_scores_final / stage2_scores_final / completeness_avg /
      coordination_avg / release_angle_final / knee_power_final / release_height_final
  - score_data（每次投篮得分）：
      score / stage1_scores / stage2_scores / completeness / coordination /
      release_angle / knee_power / release_height
  - pose_ext：每次投篮 data.json 里的 list_pose 骨架点数据（frame_id + keypoints p0~p16）
  - detail_number / video_number：从 1 开始自增
  - local_file_raw / local_file_ai：视频文件名（相对 videos 目录）
  - video_start_time / video_end_time：从视频文件名时间戳反解

依赖：os / json / re / config
"""

import json
import logging
import os
import re

logger = logging.getLogger("basketball_scoring")


# ----------------------------------------------------------------------
# 字段映射：项目 scores（12 项） -> 接口字段
# ----------------------------------------------------------------------
def map_final_data(scores):
    """scores dict -> final_data dict（字段重命名）。"""
    if not scores:
        return {}
    return {
        "final_score": scores.get("final_score"),
        "stage1_scores_final": scores.get("stage1_dtw"),
        "stage2_scores_final": scores.get("stage2_dtw"),
        "completeness_avg": scores.get("completeness"),
        "coordination_avg": scores.get("coordination"),
        "release_angle_final": scores.get("release_angle"),
        "knee_power_final": scores.get("knee_power"),
        "release_height_final": scores.get("height"),
    }


def map_score_data(scores):
    """scores dict -> score_data dict（字段重命名）。"""
    if not scores:
        return {}
    return {
        "score": scores.get("final_score"),
        "stage1_scores": scores.get("stage1_dtw"),
        "stage2_scores": scores.get("stage2_dtw"),
        "completeness": scores.get("completeness"),
        "coordination": scores.get("coordination"),
        "release_angle": scores.get("release_angle"),
        "knee_power": scores.get("knee_power"),
        "release_height": scores.get("height"),
    }


# ----------------------------------------------------------------------
# 视频文件名 -> 起止时间反解
# ----------------------------------------------------------------------
_VIDEO_TS_RE = re.compile(
    r"^\d{2}-(\d{8}_\d{6})-(\d{8}_\d{6})_(?:raw|ai)\.mp4$")


def _fmt_token(tok):
    """把 '20260911_114007' 格式化为 '2026-09-11 11:40:07'；失败返回 None。"""
    try:
        return "%s-%s-%s %s:%s:%s" % (
            tok[0:4], tok[4:6], tok[6:8], tok[9:11], tok[11:13], tok[13:15])
    except Exception:
        return None


def parse_video_times(filename):
    """从视频文件名反解 (video_start_time, video_end_time)。

    文件名形如 01-20260911_114007-20260911_114007_raw.mp4，
    解析出 "YYYY-MM-DD HH:MM:SS" 形式的起止时间；解析失败返回 (None, None)。
    """
    if not filename:
        return None, None
    m = _VIDEO_TS_RE.match(os.path.basename(filename))
    if not m:
        return None, None
    return _fmt_token(m.group(1)), _fmt_token(m.group(2))


# ----------------------------------------------------------------------
# 数据装载
# ----------------------------------------------------------------------
def _load_shot_data_json(shot_dir):
    """读取投篮 data.json；不存在/损坏返回 None。"""
    if not shot_dir:
        return None
    path = os.path.join(shot_dir, "data.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_exercise_details(shots):
    """由内存 shot 列表组装 exercise_details（pose_ext 从 data.json 读 list_pose）。

    shots: self.inference.results（每个含 scores / shot_idx / save_data /
           player_confidence_avg / basketball_confidence_avg 等）。
    返回 list[dict]；detail_number 从 1 自增（取 shot_idx）。
    """
    details = []
    for i, shot in enumerate(shots or []):
        sd = shot.get("save_data") or {}
        shot_dir = sd.get("shot_dir")
        shot_idx = shot.get("shot_idx") if shot.get("shot_idx") is not None \
            else sd.get("shot_idx")
        # detail_number 从 1 自增；shot_idx 缺失时用列表顺序兜底
        detail_number = int(shot_idx) if shot_idx is not None else (i + 1)
        scores = shot.get("scores") or {}
        # pose_ext = 该投 data.json 的 list_pose（骨架点数据）
        data_json = _load_shot_data_json(shot_dir)
        pose_ext = (data_json or {}).get("list_pose") or []
        details.append({
            "detail_number": detail_number,
            "shot_start_time": shot.get("start_time_str"),
            "shot_end_time": shot.get("end_time_str"),
            "player_confidence_avg": shot.get("player_confidence_avg", 0.0),
            "basketball_confidence_avg": shot.get("basketball_confidence_avg", 0.0),
            "pose_ext": pose_ext,
            "score_data": map_score_data(scores),
        })
    return details


def build_video_details(videos_dir):
    """扫描会话 videos 目录，组装 video_details（raw/ai 按编号配对，从 1 自增）。

    返回 list[dict]；无视频返回 []。
    """
    videos = []
    if not videos_dir or not os.path.isdir(videos_dir):
        return videos
    pairs = {}
    for name in sorted(os.listdir(videos_dir)):
        if not name.lower().endswith(".mp4"):
            continue
        m = re.match(r"^(\d{2})-.*_(raw|ai)\.mp4$", name)
        if not m:
            continue
        num = int(m.group(1))
        kind = m.group(2)
        pairs.setdefault(num, {})[kind] = name
    for num in sorted(pairs):
        raw_name = pairs[num].get("raw")
        ai_name = pairs[num].get("ai")
        if raw_name is None and ai_name is None:
            continue
        anchor = raw_name or ai_name
        v_start, v_end = parse_video_times(anchor)
        videos.append({
            "video_number": num,
            "video_start_time": v_start,
            "video_end_time": v_end,
            "local_file_raw": raw_name,
            "local_file_ai": ai_name,
        })
    return videos


# ----------------------------------------------------------------------
# 会话聚合
# ----------------------------------------------------------------------
# final_data 各字段 -> 项目 scores 源 key（求均值）
_FINAL_DIM_MAP = {
    "final_score": "final_score",
    "stage1_scores_final": "stage1_dtw",
    "stage2_scores_final": "stage2_dtw",
    "completeness_avg": "completeness",
    "coordination_avg": "coordination",
    "release_angle_final": "release_angle",
    "knee_power_final": "knee_power",
    "release_height_final": "height",
}


def _aggregate_final_data(shots):
    """聚合会话总得分：每个维度取所有投篮的均值，无投篮返回全 None。"""
    acc = {k: [] for k in _FINAL_DIM_MAP}
    for shot in shots or []:
        s = shot.get("scores") or {}
        for out_k, in_k in _FINAL_DIM_MAP.items():
            v = s.get(in_k)
            if v is not None:
                acc[out_k].append(float(v))
    result = {}
    for out_k in _FINAL_DIM_MAP:
        vals = acc[out_k]
        result[out_k] = round(sum(vals) / len(vals), 2) if vals else None
    return result


def _calc_total_duration(session_meta, shots):
    """运动总时长（秒，int）：优先 end_epoch - start_epoch，兜底用投篮起止。"""
    start = session_meta.get("start_epoch")
    end = session_meta.get("end_epoch")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) \
            and end > start:
        return int(round(end - start))
    # 兜底：最后一次投篮结束时间 - 会话开始时间
    if isinstance(start, (int, float)) and shots:
        last_end = None
        for s in shots:
            e = s.get("end_time")
            if isinstance(e, (int, float)):
                last_end = e if last_end is None else max(last_end, e)
        if last_end and last_end > start:
            return int(round(last_end - start))
    return 0


# ----------------------------------------------------------------------
# 顶层组装
# ----------------------------------------------------------------------
def build_exercise_record_payload(order_id, user_id, session_meta,
                                  shots, images_dir, videos_dir):
    """组装 addAlgorithm 完整请求体（不含 token，token 由 upload_session 注入）。

    session_meta: dict（start_time / end_time / start_epoch / end_epoch / user_id）
    shots       : self.inference.results
    images_dir  : 会话 images 目录（读每投 data.json 的 list_pose）
    videos_dir  : 会话 videos 目录（扫描 raw/ai 视频）
    返回 dict。
    """
    details = build_exercise_details(shots)
    final_data = _aggregate_final_data(shots)
    total_duration = _calc_total_duration(session_meta, shots)
    exercise_record = {
        "start_time": session_meta.get("start_time"),
        "end_time": session_meta.get("end_time"),
        "total_duration": total_duration,
        "total_shot_count": len(details),
        "final_data": final_data,
    }
    return {
        "order_id": order_id,
        "user_id": user_id,
        "exercise_record": exercise_record,
        "exercise_details": details,
        "video_details": build_video_details(videos_dir),
    }
