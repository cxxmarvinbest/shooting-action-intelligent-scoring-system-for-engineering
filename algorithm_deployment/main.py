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
  python3 main.py <测试视频路径> --no-llm          # 离线调试，跳过豆包评语

依赖：pipeline（VideoAnalyzer）、scoring（ScoringEngine）、llm_api（LLMCoach）
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
    "knee_power": "屈膝发力与爆发性",
    "release_angle": "出手角度",
}


def build_report_text(result):
    """根据结果字典拼装一份可读的纯文本评分报告"""
    s = result["scores"]
    lines = [
        "=" * 60,
        "投篮动作智能评分报告",
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
        f"  综合总得分              : {s['final_score']:.1f} / 100",
        f"  阶段1（准备-下蹲）DTW   : {s['stage1_dtw']:.1f} / 100（平均距离 {s['stage1_dtw_deg']:.3f}）",
        f"  阶段2（蹬伸-出手）DTW   : {s['stage2_dtw']:.1f} / 100（平均距离 {s['stage2_dtw_deg']:.3f}）",
        f"  核心环节技术完整度      : {s['completeness']:.1f} %",
        f"  动力链协同与发力节奏    : {s['coordination']:.1f} / 100",
        f"  屈膝发力与爆发性        : {s['knee_power']:.1f} / 100",
        f"  出手角度                : {s['release_angle']:.1f} / 100",
        f"  出手高度（相对值）      : {s['test_rel_height']:.2f}"
        f"（标准参考 {s['avg_std_height']:.2f}，得分 {s['height']:.1f} / 100）",
        "",
        "【逐模块原始报告】",
    ]
    for key, label in _REPORT_LABELS.items():
        lines.append("")
        lines.append(f"---- {label} ----")
        lines.append(result["reports"][key])
    lines += [
        "",
        "【AI 大模型（豆包）智能教练评语】",
        result["ai_report"],
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def run_scoring(test_video_path, standard_videos, out_dir, skip_llm=False):
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

    # ── 处理标准视频库 ──
    std_seqs1, std_seqs2, std_heights = [], [], []
    for i, path in enumerate(standard_videos):
        s1, s2, _, _, rel_h, _ = analyzer.process_video(path.strip(), save_visuals=False)
        if s1 is not None and len(s1) > 2:
            std_seqs1.append(s1)
        if s2 is not None and len(s2) > 2:
            std_seqs2.append(s2)
        if rel_h > 0:
            std_heights.append(rel_h)
        logger.info("标准视频 %d/%d 处理完成: %s",
                    i + 1, len(standard_videos), os.path.basename(path.strip()))

    avg_std_height = sum(std_heights) / len(std_heights) if std_heights else 0.5
    result["avg_std_height"] = round(float(avg_std_height), 4)

    if not std_seqs1 or not std_seqs2:
        raise ValueError("标准视频库解析失败，无法提取两段动作特征！")

    # ── 冠军样本选择（DTW 中位样本）──
    champ1 = ScoringEngine.select_champion(std_seqs1)
    champ2 = ScoringEngine.select_champion(std_seqs2)
    logger.info("冠军样本选择完成")

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
    deg1, score1 = ScoringEngine.compute_dtw_distance(champ1, test_s1)
    deg2, score2 = ScoringEngine.compute_dtw_distance(champ2, test_s2)
    coord_score, coord_report = ScoringEngine.compute_coordination(test_metrics)
    video_fps = getattr(analyzer, 'current_fps', 30.0)
    knee_score, knee_report = ScoringEngine.compute_knee_power(test_metrics, fps=video_fps)
    release_score, release_report = ScoringEngine.compute_release_angle(test_metrics)
    completeness_score, completeness_report = ScoringEngine.compute_completeness(
        test_metrics, fps=video_fps)

    final_score = (score1 + score2) / 2.0

    # ── 汇总日志 ──
    logger.info("-" * 60)
    logger.info("【评分结果汇总】")
    logger.info("综合总得分            : %.1f / 100", final_score)
    logger.info("阶段1（准备-下蹲）DTW : %.1f / 100（平均距离 %.3f）", score1, deg1)
    logger.info("阶段2（蹬伸-出手）DTW : %.1f / 100（平均距离 %.3f）", score2, deg2)
    logger.info("核心环节技术完整度    : %.1f %%", completeness_score)
    logger.info("动力链协同与发力节奏  : %.1f / 100", coord_score)
    logger.info("屈膝发力与爆发性      : %.1f / 100", knee_score)
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
    else:
        coach = LLMCoach()
        ai_report = coach.generate_report(score1, score2, completeness_score,
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


def main():
    parser = argparse.ArgumentParser(description="投篮动作智能评分系统（RK3588 无界面版）")
    parser.add_argument("video", help="测试视频路径")
    parser.add_argument("--std-dir", default=Config.STANDARD_VIDEO_DIR, help="标准视频目录")
    parser.add_argument("--out-dir", default=Config.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--log-dir", default=None, help="日志目录（默认=输出目录）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 评语（离线调试）")
    args = parser.parse_args()

    log_dir = args.log_dir or args.out_dir
    log_file = setup_logging(log_dir)
    logger.info("日志文件: %s", log_file)

    try:
        standard_videos = list_standard_videos(args.std_dir)
        if not standard_videos:
            logger.warning("标准视频目录 %s 下未找到 shot_*.mp4", args.std_dir)
        run_scoring(args.video, standard_videos, args.out_dir, skip_llm=args.no_llm)
        logger.info("=" * 60)
        logger.info("评分流程结束")
    except Exception:
        logger.error("处理失败:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
