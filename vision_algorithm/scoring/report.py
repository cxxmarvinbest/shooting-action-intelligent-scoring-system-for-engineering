# -*- coding: utf-8 -*-
"""
评分流程工具（scoring/report）
================================
提供「单投评分 + 报告生成」的共享函数，供离线入口与实时 controller 复用：
  - html_to_text        HTML 报告转纯文本
  - build_report_text   组装可读的纯文本评分报告
  - score_one_shot      对单次投篮动作段计算各模块得分并加权叠加

依赖：re / html / ScoringEngine / Config / LLMCoach
"""

import re
from html import unescape

from config import Config
from common.exceptions import ScoringError
from vision_algorithm.scoring.scoring_engine import ScoringEngine

_REPORT_LABELS = {
    "completeness": "核心环节技术完整度",
    "coordination": "动力链协同与发力节奏",
    "knee_power": "屈髋屈膝发力与爆发性",
    "release_angle": "出手角度",
}


def html_to_text(s):
    """把评分模块返回的 HTML 报告转成适合日志/文本文件的纯文本"""
    if not s:
        return ""
    s = re.sub(r'<br\s*/?>', '\n', s)
    s = re.sub(r'</tr>', '\n', s)
    s = re.sub(r'</t[dh]>', '  ', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = unescape(s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s.strip()


def build_report_text(result):
    """根据结果字典拼装一份可读的纯文本评分报告"""
    s = result["scores"]
    shot_title = f"（第 {result['shot_idx']} 投）" if result.get("shot_idx") else ""
    lines = [
        "=" * 60,
        f"投篮动作智能评分报告{shot_title}",
        "=" * 60,
        f"测试视频      : {result['test_video']}",
        f"输出目录      : {result['output_dir']}",
        f"标准视频数量  : {result['standard_video_count']}",
        "",
        "【输出产物】",
        f"  分段视频1（准备-下蹲）: {result['clip1_video']}",
        f"  分段视频2（蹬伸-出手）: {result['clip2_video']}",
        f"  逐帧图片目录           : {result['frames_dir']}",
        "",
        "【投篮时间】",
        f"  开始时间 : {result.get('start_time_str', 'N/A')}",
        f"  结束时间 : {result.get('end_time_str', 'N/A')}",
        f"  本次用时 : {result.get('duration_str', 'N/A')}",
        "",
        "【评分结果】",
        f"  综合总得分           : {s['final_score']:.1f} / 100",
        f"  阶段1（准备-下蹲）    : {s['stage1_dtw']:.1f} / 100",
        f"  阶段2（蹬伸-出手）    : {s['stage2_dtw']:.1f} / 100",
        f"  核心环节技术完整度     : {s['completeness']:.1f} %",
        f"  动力链协同与发力节奏    : {s['coordination']:.1f} / 100",
        f"  屈髋屈膝发力与爆发性    : {s['knee_power']:.1f} / 100",
        f"  出手角度              : {s['release_angle']:.1f} / 100",
        f"  出手高度（相对值）      : {s['height']:.1f} / 100"
        "",
        "【逐模块原始报告】",
    ]
    for key, label in _REPORT_LABELS.items():
        lines.append("")
        lines.append(f"---- {label} ----")
        lines.append(result["reports"][key])
    lines += [
        "",
        "【AI 大模型智能教练评语】" if "本地自动评语" not in result["ai_report"]
        else "【本地自动评语（离线 / 接口不可用）】",
        result["ai_report"],
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def score_one_shot(champ1, champ2, avg_std_height, test_s1, test_s2,
                   test_rel_h, test_metrics, video_fps=30.0):
    """对单次投篮动作段计算各模块得分并加权叠加，返回 (scores, reports)。

    供离线评分与实时评分复用，保证两者评分口径完全一致。
    异常捕获：输入数据异常（frame_metrics 为空 / 数组长度不一致）统一抛 ScoringError。
    """
    if test_metrics is None or len(test_metrics) == 0:
        raise ScoringError(
            "评分输入 frame_metrics 为空（输入数据异常）", kind="scoring")
    if len(test_s1) < 2 or len(test_s2) < 2:
        raise ScoringError(
            f"角度序列过短（s1={len(test_s1)} 帧, s2={len(test_s2)} 帧），"
            "无法进行 DTW 比对（数组长度不一致）", kind="scoring")

    height_score = ScoringEngine.compute_height_score(test_rel_h, avg_std_height)
    deg1, dtw_score1 = ScoringEngine.compute_dtw_distance(champ1, test_s1)
    deg2, dtw_score2 = ScoringEngine.compute_dtw_distance(champ2, test_s2)
    coord_score, coord_report = ScoringEngine.compute_coordination(test_metrics)
    knee_score, knee_report = ScoringEngine.compute_knee_power(test_metrics, fps=video_fps)
    release_score, release_report = ScoringEngine.compute_release_angle(test_metrics)
    completeness_score, completeness_report = ScoringEngine.compute_completeness(
        test_metrics, fps=video_fps)

    module_scores = {
        "stage1_dtw": dtw_score1,
        "stage2_dtw": dtw_score2,
        "completeness": completeness_score,
        "coordination": coord_score,
        "knee_power": knee_score,
        "release_angle": release_score,
        "height": height_score,
    }
    reports = {
        "completeness": (completeness_score, completeness_report),
        "coordination": (coord_score, coord_report),
        "knee_power": (knee_score, knee_report),
        "release_angle": (release_score, release_report),
    }

    aux_scores = {k: v for k, v in module_scores.items()
                  if k not in ("stage1_dtw", "stage2_dtw")}
    aux_weighted = ScoringEngine.combine_scores(aux_scores, Config.SCORE_WEIGHTS)
    ratio = Config.PHASE_DTW_RATIO
    score1 = ratio * dtw_score1 + (1.0 - ratio) * aux_weighted
    score2 = ratio * dtw_score2 + (1.0 - ratio) * aux_weighted
    final_score = ScoringEngine.combine_scores(module_scores, Config.SCORE_WEIGHTS)

    scores = {
        "final_score": round(final_score, 2),
        "stage1_dtw": round(score1, 2),
        "stage1_dtw_deg": round(deg1, 4),
        "stage2_dtw": round(score2, 2),
        "stage2_dtw_deg": round(deg2, 4),
        "completeness": round(completeness_score, 2),
        "coordination": round(coord_score, 2),
        "knee_power": round(knee_score, 2),
        "release_angle": round(release_score, 2),
        "height": round(height_score, 2),
        "test_rel_height": round(test_rel_h, 4),
        "avg_std_height": round(float(avg_std_height), 4),
    }
    return scores, reports
