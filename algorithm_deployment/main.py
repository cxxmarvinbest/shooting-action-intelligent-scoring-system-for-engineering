# -*- coding: utf-8 -*-
"""
主程序模块（main）—— RK3588 无界面 / 日志版
=============================================
职责：
  1. 日志初始化（控制台 + 文件双输出）
  2. 串联「检测跟踪 → 动作分段 → 各项评分 → LLM 评语」全流程
  3. 把分段视频、逐帧图片、每项评分及其原始报告、LLM 评语统一写入
     日志文件、纯文本报告、JSON 结构化结果（供后续 HTTP API 读取返回给 APP）

用法：
  python3 main.py <测试视频路径>
  python3 main.py <测试视频路径> --std-dir <标准视频目录> --out-dir <输出目录>
  python3 main.py <测试视频路径> --no-llm          # 离线调试，跳过豆包评语（不生成任何评语）
  python3 main.py <测试视频路径> --offline         # 离线模式：跳过豆包 API，用本地规则生成评语
  python3 main.py <测试视频路径> --multi            # 多投篮模式：环形缓存逐投切分并分别评分
  LQ_OFFLINE=1 python3 main.py <测试视频路径>       # 环境变量开启离线模式（部署机免改代码）

依赖：pipeline（VideoAnalyzer）、standard_lib（StandardLibrary）、scoring（ScoringEngine）、llm_api（LLMCoach）
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from html import unescape

from config import Config
from pipeline import VideoAnalyzer
from scoring import ScoringEngine
from llm_api import LLMCoach
from standard_lib import StandardLibrary

logger = logging.getLogger("basketball_scoring")


def list_standard_videos(std_dir):
    """扫描标准视频目录下的 shot_*.mp4，按序号排序返回全路径列表"""
    files = [f for f in os.listdir(std_dir)
             if re.fullmatch(r"shot_\d+\.mp4", f, re.IGNORECASE)]
    files.sort(key=lambda f: int(re.search(r"\d+", f).group()))
    return [os.path.join(std_dir, f) for f in files]


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


def setup_logging(log_dir):
    """初始化日志：控制台 + 文件，返回日志文件路径"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, time.strftime("scoring_%Y%m%d_%H%M%S.log"))
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    logger.setLevel(logging.INFO)
    # 避免重复注册 handler（作为模块被多次调用时）
    logger.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_file


_REPORT_LABELS = {
    "completeness": "核心环节技术完整度",
    "coordination": "动力链协同与发力节奏",
    "knee_power": "屈髋屈膝发力与爆发性",
    "release_angle": "出手角度",
}


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

    与 run_scoring 中的评分逻辑完全一致，供多投篮逐投评分复用。
    """
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


def run_scoring(test_video_path, standard_videos, out_dir, skip_llm=False, offline=False):
    """
    执行完整评分流程，返回结果字典。

    结果字典包含：输出产物路径、各模块评分、各模块原始报告、LLM 评语。
    """
    result = {
        "test_video": test_video_path,
        "output_dir": out_dir,
        "standard_video_count": len(standard_videos),
    }

    logger.info("=" * 60)
    logger.info("开始投篮动作评分（无界面/日志模式）")
    logger.info("测试视频: %s", test_video_path)
    logger.info("标准视频数量: %d", len(standard_videos))
    logger.info("输出目录: %s", out_dir)

    # ── 模块一：加载 RKNN 检测/姿态模型 ──
    logger.info("加载 RKNN 检测/姿态模型 ...")
    analyzer = VideoAnalyzer()
    analyzer.load_models()
    logger.info("模型加载完成")

    # ── 预加载标准视频库特征（只跑一次推理，缓存冠军样本 + 平均出手高度）──
    std_lib = StandardLibrary(analyzer)
    std_lib.build(standard_videos, cache_path=Config.STANDARD_CACHE_PATH)
    champ1, champ2 = std_lib.champ1, std_lib.champ2
    avg_std_height = std_lib.avg_std_height
    result["avg_std_height"] = round(float(avg_std_height), 4)

    # ── 处理测试视频（含可视化输出：分段视频 + 逐帧图）──
    logger.info("处理测试视频并输出分段视频/逐帧图 ...")
    test_s1, test_s2, out_v1, out_v2, test_rel_h, test_metrics = analyzer.process_video(
        test_video_path, save_visuals=True, out_dir=out_dir)

    if test_s1 is None or test_s2 is None:
        raise ValueError("测试视频未检测到完整的下蹲和出手动作！")

    result["clip1_video"] = out_v1
    result["clip2_video"] = out_v2
    result["frames_dir"] = os.path.join(out_dir, "frames")

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

    # ── 加权叠加：DTW 评分与其它模块评分按权重融合，覆盖综合总得分 / 阶段1 / 阶段2 ──
    # 各模块得分（0~100），键名与 Config.SCORE_WEIGHTS 一一对应
    module_scores = {
        "stage1_dtw": dtw_score1,
        "stage2_dtw": dtw_score2,
        "completeness": completeness_score,
        "coordination": coord_score,
        "knee_power": knee_score,
        "release_angle": release_score,
        "height": height_score,
    }
    # 其它模块（非 DTW）的加权均分，用于阶段1/阶段2 的加权叠加
    aux_scores = {k: v for k, v in module_scores.items()
                  if k not in ("stage1_dtw", "stage2_dtw")}
    aux_weighted = ScoringEngine.combine_scores(aux_scores, Config.SCORE_WEIGHTS)

    # 阶段1 / 阶段2：DTW 自身按 PHASE_DTW_RATIO 占比，其余由其它模块加权均分补足
    ratio = Config.PHASE_DTW_RATIO
    score1 = ratio * dtw_score1 + (1.0 - ratio) * aux_weighted
    score2 = ratio * dtw_score2 + (1.0 - ratio) * aux_weighted
    # 综合总得分：全部模块按权重加权叠加（权重自动归一化）
    final_score = ScoringEngine.combine_scores(module_scores, Config.SCORE_WEIGHTS)

    # ── 汇总日志 ──
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

    # ── 逐模块原始报告 ──
    reports = {
        "completeness": (completeness_score, completeness_report),
        "coordination": (coord_score, coord_report),
        "knee_power": (knee_score, knee_report),
        "release_angle": (release_score, release_report),
    }
    for name, (score, report) in reports.items():
        logger.info("\n---- %s（%.1f）----\n%s", _REPORT_LABELS[name], score, html_to_text(report))

    # ── 调用豆包 API 生成 AI 教练评语 ──
    logger.info("-" * 60)
    if skip_llm:
        ai_report = "（已跳过 LLM 评语）"
        logger.info("【AI 大模型评语】已按 --no-llm 跳过")
    elif offline or Config.OFFLINE_MODE:
        # 离线模式：完全不联网，用本地规则生成评语，断网测试也不会报错
        coach = LLMCoach(offline=True)
        ai_report = coach.generate_report(dtw_score1, dtw_score2, completeness_score,
                                          coord_score, knee_score, release_score)
        logger.info("【AI 大模型评语】离线模式（本地生成，未调用豆包）\n%s", ai_report)
    else:
        coach = LLMCoach()  # 在线模式：断网/超时会自动降级为本地评语，不抛出
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

    # ── 写纯文本报告 ──
    report_path = os.path.join(out_dir, time.strftime("report_%Y%m%d_%H%M%S.txt"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report_text(result))
    logger.info("纯文本报告已写入: %s", report_path)

    # ── 写 JSON 结构化结果（供 HTTP API 读取）──
    json_path = os.path.join(out_dir, time.strftime("result_%Y%m%d_%H%M%S.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("JSON 结构化结果已写入: %s", json_path)

    return result


def run_scoring_multi(test_video_path, standard_videos, out_dir,
                      skip_llm=False, offline=False):
    """多投篮完整视频：关键点环形缓存逐投切分并评分，每投独立报告 + 汇总清单。

    返回结果字典，含 shots 列表（每投的完整评分结果）。
    """
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

        # LLM 评语（与单投篮口径一致）
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
        logger.info("第 %d 次投篮评分完成：综合 %.1f / 100（起点=%d 出手=%d）",
                    shot_idx, scores["final_score"],
                    seg['start_idx'], seg['release_idx'])

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

    return result


def main():
    parser = argparse.ArgumentParser(description="投篮动作智能评分系统（RK3588 无界面版）")
    parser.add_argument("video", help="测试视频路径")
    parser.add_argument("--std-dir", default=Config.STANDARD_VIDEO_DIR, help="标准视频目录")
    parser.add_argument("--out-dir", default=Config.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--log-dir", default=None, help="日志目录（默认=输出目录）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 评语（不生成任何评语）")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：跳过豆包 API，用本地规则生成评语（断网测试用）")
    parser.add_argument("--multi", action="store_true",
                        help="多投篮模式：用关键点环形缓存逐投切分并分别评分（完整多投篮视频用）")
    args = parser.parse_args()

    log_dir = args.log_dir or args.out_dir
    log_file = setup_logging(log_dir)
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
