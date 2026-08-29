# -*- coding: utf-8 -*-
"""
离线视频评分主入口（offline_video_scoring）
=============================================
职责（主线程入口，只做「装配 + 流程编排」，不承载算法）：
  1. 日志初始化（统一走 common/logger）
  2. 串联「检测跟踪 → 动作分段 → 各项评分 → LLM 评语」全流程
  3. 把分段视频、逐帧图片、评分原始报告、LLM 评语写入日志/文本报告/JSON

用法：
  python offline_video_scoring.py <测试视频路径>
  python offline_video_scoring.py <测试视频路径> --std-dir <标准视频目录> --out-dir <输出目录>
  python offline_video_scoring.py <测试视频路径> --no-llm     # 跳过豆包评语
  python offline_video_scoring.py <测试视频路径> --offline    # 离线模式：本地规则评语
  python offline_video_scoring.py <测试视频路径> --multi      # 多投篮模式：环形缓存逐投评分
  LQ_OFFLINE=1 python offline_video_scoring.py <测试视频路径>

依赖：config / common.logger / vision_algorithm.*
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback

from config import Config
from common.logger import setup_logger
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer
from vision_algorithm.scoring.scoring_engine import ScoringEngine
from vision_algorithm.scoring.report import (
    build_report_text, html_to_text, score_one_shot)
from vision_algorithm.llm.llm_coach import LLMCoach
from vision_algorithm.standard.standard_library import (
    StandardLibrary, list_standard_videos)
from vision_algorithm.segmentation.shot_segmenter import (
    format_shot_time, format_duration)

logger = logging.getLogger("basketball_scoring")


def run_scoring(test_video_path, standard_videos, out_dir, skip_llm=False, offline=False):
    """执行完整评分流程，返回结果字典。"""
    result = {
        "test_video": test_video_path,
        "output_dir": out_dir,
        "standard_video_count": len(standard_videos),
    }

    logger.info("=" * 60)
    logger.info("开始投篮动作评分（离线模式）")
    logger.info("测试视频: %s", test_video_path)
    logger.info("标准视频数量: %d", len(standard_videos))
    logger.info("输出目录: %s", out_dir)

    # ── 加载 RKNN 检测/姿态模型 ──
    logger.info("加载 RKNN 检测/姿态模型 ...")
    analyzer = VideoAnalyzer()
    analyzer.load_models()
    logger.info("模型加载完成")

    # ── 预加载标准视频库特征 ──
    std_lib = StandardLibrary(analyzer)
    std_lib.build(standard_videos, cache_path=Config.STANDARD_CACHE_PATH)
    champ1, champ2 = std_lib.champ1, std_lib.champ2
    avg_std_height = std_lib.avg_std_height
    result["avg_std_height"] = round(float(avg_std_height), 4)

    # ── 处理测试视频（含可视化输出）──
    logger.info("处理测试视频并输出分段视频/逐帧图 ...")
    test_s1, test_s2, out_v1, out_v2, test_rel_h, test_metrics = analyzer.process_video(
        test_video_path, save_visuals=True, out_dir=out_dir)

    if test_s1 is None or test_s2 is None:
        raise ValueError("测试视频未检测到完整的下蹲和出手动作！")

    result["clip1_video"] = out_v1
    result["clip2_video"] = out_v2
    result["frames_dir"] = os.path.join(out_dir, "frames")

    # ── 单投篮起止时间（视频内相对时间，秒）──
    # test_metrics 为一帧特征 dict 列表，每帧已带 ts；取首尾帧 ts 即得本投时间跨度。
    _t0 = test_metrics[0].get("ts") if test_metrics else None
    _t1 = test_metrics[-1].get("ts") if test_metrics else None
    _dur = (_t1 - _t0) if (_t0 is not None and _t1 is not None) else None
    result["start_time"] = _t0
    result["end_time"] = _t1
    result["duration"] = _dur
    result["start_time_str"] = format_shot_time(_t0)
    result["end_time_str"] = format_shot_time(_t1)
    result["duration_str"] = format_duration(_dur)
    logger.info("投篮起止时间（视频内相对）：开始 %s 结束 %s 用时 %s",
                result["start_time_str"], result["end_time_str"], result["duration_str"])

    # ── 各项评分 ──
    height_score = ScoringEngine.compute_height_score(test_rel_h, avg_std_height)
    deg1, dtw_score1 = ScoringEngine.compute_dtw_distance(champ1, test_s1)
    deg2, dtw_score2 = ScoringEngine.compute_dtw_distance(champ2, test_s2)
    coord_score, coord_report = ScoringEngine.compute_coordination(test_metrics)
    video_fps = getattr(analyzer, 'current_fps', 30.0)
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
    aux_scores = {k: v for k, v in module_scores.items()
                  if k not in ("stage1_dtw", "stage2_dtw")}
    aux_weighted = ScoringEngine.combine_scores(aux_scores, Config.SCORE_WEIGHTS)
    ratio = Config.PHASE_DTW_RATIO
    score1 = ratio * dtw_score1 + (1.0 - ratio) * aux_weighted
    score2 = ratio * dtw_score2 + (1.0 - ratio) * aux_weighted
    final_score = ScoringEngine.combine_scores(module_scores, Config.SCORE_WEIGHTS)

    logger.info("-" * 60)
    logger.info("【评分结果汇总】")
    logger.info("综合总得分    : %.1f / 100", final_score)
    logger.info("阶段1（准备-下蹲） : %.1f / 100（平均距离 %.3f）", dtw_score1, deg1)
    logger.info("阶段2（蹬伸-出手） : %.1f / 100（平均距离 %.3f）", dtw_score2, deg2)
    logger.info("核心环节技术完整度    : %.1f %%", completeness_score)
    logger.info("动力链协同与发力节奏  : %.1f / 100", coord_score)
    logger.info("屈髋屈膝发力与爆发性  : %.1f / 100", knee_score)
    logger.info("出手角度              : %.1f / 100", release_score)
    logger.info("出手高度（相对值）    : %.2f（标准参考 %.2f，得分 %.1f / 100）",
                test_rel_h, avg_std_height, height_score)

    reports = {
        "completeness": (completeness_score, completeness_report),
        "coordination": (coord_score, coord_report),
        "knee_power": (knee_score, knee_report),
        "release_angle": (release_score, release_report),
    }
    _REPORT_LABELS = {
        "completeness": "核心环节技术完整度",
        "coordination": "动力链协同与发力节奏",
        "knee_power": "屈髋屈膝发力与爆发性",
        "release_angle": "出手角度",
    }
    for name, (score, report) in reports.items():
        logger.info("\n---- %s（%.1f）----\n%s",
                    _REPORT_LABELS[name], score, html_to_text(report))

    # ── 豆包 API 生成 AI 教练评语 ──
    logger.info("-" * 60)
    if skip_llm:
        ai_report = "（已跳过 LLM 评语）"
        logger.info("【AI 大模型评语】已按 --no-llm 跳过")
    elif offline or Config.OFFLINE_MODE:
        coach = LLMCoach(offline=True)
        ai_report = coach.generate_report(dtw_score1, dtw_score2, completeness_score,
                                          coord_score, knee_score, release_score)
        logger.info("【AI 大模型评语】离线模式（本地生成，未调用豆包）\n%s", ai_report)
    else:
        coach = LLMCoach()
        ai_report = coach.generate_report(dtw_score1, dtw_score2, completeness_score,
                                          coord_score, knee_score, release_score)
        logger.info("【AI 大模型（豆包）智能教练评语】\n%s", ai_report)

    # ── 组装结果 ──
    result["scores"] = {
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
    result["reports"] = {k: html_to_text(v[1]) for k, v in reports.items()}
    result["ai_report"] = ai_report

    # ── 写纯文本报告 + JSON ──
    report_path = os.path.join(out_dir, time.strftime("report_%Y%m%d_%H%M%S.txt"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report_text(result))
    logger.info("纯文本报告已写入: %s", report_path)

    json_path = os.path.join(out_dir, time.strftime("result_%Y%m%d_%H%M%S.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("JSON 结构化结果已写入: %s", json_path)

    analyzer.release_models()  # 释放两个 RKNN 实例占用的 NPU 资源
    return result


def run_scoring_multi(test_video_path, standard_videos, out_dir,
                      skip_llm=False, offline=False):
    """多投篮完整视频：关键点环形缓存逐投切分并评分，每投独立报告 + 汇总清单。"""
    result = {
        "test_video": test_video_path,
        "output_dir": out_dir,
        "standard_video_count": len(standard_videos),
        "shots": [],
    }

    logger.info("=" * 60)
    logger.info("开始多投篮视频评分（关键点环形缓存逐投切分）")
    logger.info("测试视频: %s", test_video_path)
    logger.info("标准视频数量: %d", len(standard_videos))
    logger.info("输出目录: %s", out_dir)

    analyzer = VideoAnalyzer()
    analyzer.load_models()

    std_lib = StandardLibrary(analyzer)
    std_lib.build(standard_videos, cache_path=Config.STANDARD_CACHE_PATH)
    champ1, champ2 = std_lib.champ1, std_lib.champ2
    avg_std_height = std_lib.avg_std_height
    result["avg_std_height"] = round(float(avg_std_height), 4)

    video_fps = getattr(analyzer, 'current_fps', 30.0)
    shots = analyzer.process_video_multi(
        test_video_path, save_visuals=True, out_dir=out_dir)

    if not shots:
        raise ValueError("多投篮视频未检测到任何完整的持球→出手动作！")

    for seg in shots:
        shot_idx = seg['shot_idx']
        scores, reports = score_one_shot(
            champ1, champ2, avg_std_height,
            seg['seq1'], seg['seq2'], seg['rel_height'], seg['frame_metrics'],
            video_fps=video_fps)

        if skip_llm:
            ai_report = "（已跳过 LLM 评语）"
        elif offline or Config.OFFLINE_MODE:
            coach = LLMCoach(offline=True)
            ai_report = coach.generate_report(
                scores["stage1_dtw"], scores["stage2_dtw"],
                scores["completeness"], scores["coordination"],
                scores["knee_power"], scores["release_angle"])
        else:
            coach = LLMCoach()
            ai_report = coach.generate_report(
                scores["stage1_dtw"], scores["stage2_dtw"],
                scores["completeness"], scores["coordination"],
                scores["knee_power"], scores["release_angle"])

        shot_result = {
            "test_video": test_video_path,
            "output_dir": out_dir,
            "standard_video_count": len(standard_videos),
            "shot_idx": shot_idx,
            "start_idx": seg['start_idx'],
            "release_idx": seg['release_idx'],
            "idx_squat": seg['idx_squat'],
            "start_time": seg.get('start_time'),
            "end_time": seg.get('end_time'),
            "duration": seg.get('duration'),
            "start_time_str": format_shot_time(seg.get('start_time')),
            "end_time_str": format_shot_time(seg.get('end_time')),
            "duration_str": format_duration(seg.get('duration')),
            "clip1_video": seg.get('clip1_path'),
            "clip2_video": seg.get('clip2_path'),
            "frames_dir": os.path.join(out_dir, "frames", f"shot{shot_idx}"),
            "scores": scores,
            "reports": {k: html_to_text(v[1]) for k, v in reports.items()},
            "ai_report": ai_report,
        }

        report_path = os.path.join(out_dir, f"report_shot{shot_idx:02d}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(build_report_text(shot_result))
        json_path = os.path.join(out_dir, f"result_shot{shot_idx:02d}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(shot_result, f, ensure_ascii=False, indent=2)

        result["shots"].append(shot_result)
        logger.info(
            "第 %d 次投篮评分完成：综合 %.1f / 100（起点=%d 出手=%d，"
            "开始 %s 结束 %s 用时 %s）",
            shot_idx, scores["final_score"],
            seg['start_idx'], seg['release_idx'],
            format_shot_time(seg.get('start_time')),
            format_shot_time(seg.get('end_time')),
            format_duration(seg.get('duration')))

    summary = {
        "test_video": test_video_path,
        "output_dir": out_dir,
        "shot_count": len(shots),
        "avg_std_height": result["avg_std_height"],
        "shots": [
            {
                "shot_idx": s["shot_idx"],
                "start_idx": s["start_idx"],
                "release_idx": s["release_idx"],
                "start_time": s.get("start_time"),
                "end_time": s.get("end_time"),
                "duration": s.get("duration"),
                "final_score": s["scores"]["final_score"],
                "stage1_dtw": s["scores"]["stage1_dtw"],
                "stage2_dtw": s["scores"]["stage2_dtw"],
                "completeness": s["scores"]["completeness"],
                "coordination": s["scores"]["coordination"],
                "knee_power": s["scores"]["knee_power"],
                "release_angle": s["scores"]["release_angle"],
                "height": s["scores"]["height"],
                "clip1_video": s["clip1_video"],
                "clip2_video": s["clip2_video"],
            }
            for s in result["shots"]
        ],
    }
    summary_path = os.path.join(out_dir, "result_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("汇总清单已写入: %s", summary_path)

    analyzer.release_models()  # 释放两个 RKNN 实例占用的 NPU 资源
    return result


def main():
    parser = argparse.ArgumentParser(description="投篮动作智能评分系统（RK3588 离线版）")
    parser.add_argument("video", help="测试视频路径")
    parser.add_argument("--std-dir", default=Config.STANDARD_VIDEO_DIR, help="标准视频目录")
    parser.add_argument("--out-dir", default=Config.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--log-dir", default=None, help="日志目录（默认=输出目录）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 评语")
    parser.add_argument("--offline", action="store_true", help="离线模式：本地规则评语")
    parser.add_argument("--multi", action="store_true", help="多投篮模式：逐投切分评分")
    args = parser.parse_args()

    log_dir = args.log_dir or args.out_dir
    log_file = setup_logger(log_dir)
    logger.info("日志文件: %s", log_file)
    if args.offline or Config.OFFLINE_MODE:
        logger.info("离线模式：已开启（不调用豆包 API）")

    try:
        standard_videos = list_standard_videos(args.std_dir)
        if not standard_videos:
            logger.warning("标准视频目录 %s 下未找到 shot_*.mp4", args.std_dir)
        if args.multi:
            run_scoring_multi(args.video, standard_videos, args.out_dir,
                              skip_llm=args.no_llm, offline=args.offline)
        else:
            run_scoring(args.video, standard_videos, args.out_dir,
                        skip_llm=args.no_llm, offline=args.offline)
        logger.info("=" * 60)
        logger.info("评分流程结束")
    except Exception:
        logger.error("处理失败:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
