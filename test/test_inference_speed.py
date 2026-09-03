# -*- coding: utf-8 -*-
"""
推理速度 + 投篮切分测试（test/test_inference_speed）
====================================================
测量「目标检测 + 姿态估计」单帧总耗时（含预处理），同时跑投篮动作切分状态机，
输出：
  1. 总共检测到多少次投篮动作；
  2. 每次投篮的开始/结束时间（视频内相对「标准时间」 T+mm:ss.xxx）与用时；
  3. 单帧平均/最小/最大/中位耗时与平均 FPS，判断是否满足实时要求（< 60ms/帧）。

用法（在项目根目录运行）：
    python test/test_inference_speed.py <视频路径> [--stride 2] [--timeout 100] [--frames 200]

参数：
    --stride   隔帧采样步长（默认 2，即每 2 帧处理一次，可调 2~3）
    --timeout  单帧推理超时阈值（ms），超过则视为超时跳过、处理下一帧（默认 100）
    --frames   最多处理的帧数（默认 200）

说明：
  - 用 cv2 软解视频取帧，每帧经「长边640+黑边 letterbox + 检测(0=player,1=basketball)
    + player 原图抠图 保持宽高比 letterbox 320x320 姿态估计」流水线计时；
  - 检测模型输出 player/ball 两类；姿态估计只对 player 裁剪图跑；
  - 每帧打印人/球检测状态 + 姿态是否完成 + 单帧耗时；
  - 投篮起止「标准时间」= 视频内相对时间（T+mm:ss.xxx），按原始帧号 / 真实帧率换算，
    与 offline_video_scoring.py --multi 的口径一致。

依赖：numpy / cv2 / config / vision_algorithm
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# 把项目根目录加入 sys.path，保证可从任意位置运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402

from config import Config  # noqa: E402
from common.logger import setup_logger  # noqa: E402
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer  # noqa: E402
from vision_algorithm.segmentation.shot_fsm import ShotFSM  # noqa: E402
from vision_algorithm.segmentation.shot_segmenter import (  # noqa: E402
    format_shot_time, format_duration)

logger = logging.getLogger("basketball_scoring")


def main():
    parser = argparse.ArgumentParser(description="投篮评分系统推理速度 + 投篮切分测试")
    parser.add_argument("video", help="测试视频路径")
    parser.add_argument("--stride", type=int, default=2, help="隔帧采样步长（默认 2）")
    parser.add_argument("--timeout", type=int, default=100, help="单帧超时阈值 ms（默认 100）")
    parser.add_argument("--frames", type=int, default=0, help="最多处理帧数（0=不限制，处理整段视频）")
    parser.add_argument("--log-dir", default=None, help="日志目录（默认 logs）")
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(PROJECT_ROOT, "logs")
    setup_logger(log_dir)
    logger.info("=" * 60)
    logger.info("推理速度 + 投篮切分测试开始")
    logger.info("视频: %s", args.video)
    logger.info("采样步长=%d, 超时阈值=%dms, 最多处理=%d 帧",
                args.stride, args.timeout, args.frames)

    if not os.path.exists(args.video):
        logger.error("视频文件不存在: %s", args.video)
        sys.exit(1)

    # 加载检测 + 姿态模型
    logger.info("加载 RKNN 检测/姿态模型 ...")
    analyzer = VideoAnalyzer()
    analyzer.load_models()
    logger.info("模型加载完成")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("无法打开视频: %s", args.video)
        sys.exit(1)

    stride = max(1, args.stride)
    timeout = max(1, args.timeout)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    logger.info("视频帧率=%.1f, 总帧≈%d, 分辨率=%dx%d（标准时间按原始帧号/帧率换算）",
                fps, total_frames, vid_w, vid_h)

    # 投篮切分状态机（与 process_video_multi 口径一致）
    segmenter = ShotFSM()
    shots = []

    times = []        # 正常帧耗时（ms）
    timeout_count = 0  # 超时帧数
    person_count = 0   # 检测到人的帧数
    ball_count = 0     # 检测到球的帧数
    # ── 持球诊断：球+手腕同时可见时的球心距手腕，及判为持球的帧数 ──
    ball_wrist_dists = []   # 球心距手腕（球与手腕同时可见时）
    held_count = 0          # 判为持球（dist < 0.35*player_h）的帧数
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        if args.frames > 0 and processed >= args.frames:
            break

        t0 = time.perf_counter()
        try:
            # 标准时间：视频内相对时间 = 原始帧号 / 真实帧率（与离线评分口径一致）
            fd = analyzer._extract_frame_metrics(
                frame, frame_idx, ts=frame_idx / fps)
        except Exception as e:
            logger.error("帧 %d 推理异常（%s: %s）", frame_idx, type(e).__name__, e)
            frame_idx += 1
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # 人/球检测状态（每帧输出）
        person_str = "已检测到人" if fd.get('player_box') is not None else "未检测到人"
        ball_str = "已检测到球" if fd.get('ball_boxes') else "未检测到球"
        kpts = fd.get('kpts')
        if kpts is not None and len(kpts) > 0:
            # 可见关键点 = 坐标非 (0,0) 的点
            n_visible = int((np.asarray(kpts)[:, 0] > 0).sum())
            pose_str = f"姿态点 {n_visible}/{len(kpts)}"
        else:
            pose_str = "姿态点 0/17"
        if fd.get('player_box') is not None:
            person_count += 1
        if fd.get('ball_boxes'):
            ball_count += 1

        # 持球诊断：统计球心距手腕（与 ShotSegmenter._is_ball_held 同口径）
        balls = fd.get('ball_boxes') or []
        wrist_x = fd.get('wrist_x')
        wrist_y = fd.get('wrist_y')
        player_h = fd.get('player_h')
        if balls and wrist_x is not None and wrist_y is not None \
                and wrist_x > 0 and wrist_y > 0 and player_h and player_h > 0:
            bx1, by1, bx2, by2 = balls[0]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = ((bcx - wrist_x) ** 2 + (bcy - wrist_y) ** 2) ** 0.5
            ball_wrist_dists.append(dist)
            if dist < Config.BALL_WRIST_DIST_RATIO * player_h:
                held_count += 1

        if dt_ms > timeout:
            timeout_count += 1
            logger.warning("帧 %d | %s | %s | %s | 耗时 %.1f ms > %d ms（超时，跳过）",
                           frame_idx, person_str, ball_str, pose_str, dt_ms, timeout)
        else:
            times.append(dt_ms)
            logger.info("帧 %d | %s | %s | %s | 耗时 %.1f ms",
                        frame_idx, person_str, ball_str, pose_str, dt_ms)

        # 喂给投篮切分状态机，检测完整投篮动作
        seg = segmenter.feed(fd).get('shot_event')
        if seg is not None:
            if len(seg['frame_metrics']) < Config.MIN_SHOT_FRAMES:
                logger.warning("丢弃过短段: 起点=%d, 出手=%d, 段长=%d 帧 < %d",
                               seg['start_idx'], seg['release_idx'],
                               len(seg['frame_metrics']), Config.MIN_SHOT_FRAMES)
            elif not seg.get('has_squat'):
                logger.warning("丢弃无真实下蹲的误检段: 起点=%d, 出手=%d, 段长=%d 帧",
                               seg['start_idx'], seg['release_idx'],
                               len(seg['frame_metrics']))
            else:
                shots.append(seg)
                logger.info("检测到第 %d 次投篮: 起点=%d, 出手=%d, 段长=%d 帧",
                            seg['shot_idx'], seg['start_idx'], seg['release_idx'],
                            len(seg['frame_metrics']))

        processed += 1
        frame_idx += 1

    segmenter.finalize()
    cap.release()
    analyzer.release_models()  # 显式释放两个 RKNN 实例占用的 NPU 资源

    # ── 人/球检测统计 ──
    logger.info("=" * 60)
    logger.info("【人/球检测统计】")
    logger.info("  检测到人 : %d 帧", person_count)
    logger.info("  未检测到人 : %d 帧", processed - person_count)
    logger.info("  检测到球 : %d 帧", ball_count)
    logger.info("  未检测到球 : %d 帧", processed - ball_count)

    # ── 持球诊断 ──
    logger.info("=" * 60)
    logger.info("【持球诊断】（球+手腕同时可见时的球心距手腕）")
    if ball_wrist_dists:
        arr = np.array(ball_wrist_dists)
        logger.info("  球+手腕同时可见 : %d 帧", len(arr))
        logger.info("  判为持球(<0.35*player_h) : %d 帧", held_count)
        logger.info("  球心距手腕 min/median/max : %.1f / %.1f / %.1f px",
                    arr.min(), float(np.median(arr)), arr.max())
    else:
        logger.info("  球+手腕同时可见 : 0 帧（球或手腕关键点始终缺一）")
    logger.info("  阈值口径 BALL_WRIST_DIST_RATIO=%.2f", Config.BALL_WRIST_DIST_RATIO)

    # ── 投篮结果汇总 ──
    logger.info("=" * 60)
    logger.info("【投篮切分结果】")
    logger.info("  共检测到投篮动作 : %d 次", len(shots))
    for seg in shots:
        logger.info(
            "  第 %d 次投篮: 起点帧=%d 出手帧=%d | 开始 %s | 结束 %s | 用时 %s",
            seg['shot_idx'], seg['start_idx'], seg['release_idx'],
            format_shot_time(seg.get('start_time')),
            format_shot_time(seg.get('end_time')),
            format_duration(seg.get('duration')))

    # ── 单帧耗时统计 ──
    if not times:
        logger.warning("无有效耗时样本（可能视频无帧或全部超时）")
    else:
        arr = np.array(times)
        logger.info("-" * 60)
        logger.info("【推理速度统计】（检测 + 姿态 单帧总耗时）")
        logger.info("  有效样本 : %d 帧", len(arr))
        logger.info("  平均耗时 : %.1f ms", arr.mean())
        logger.info("  最小耗时 : %.1f ms", arr.min())
        logger.info("  最大耗时 : %.1f ms", arr.max())
        logger.info("  中位耗时 : %.1f ms", float(np.median(arr)))
        logger.info("  平均 FPS : %.1f", 1000.0 / arr.mean())
        if timeout_count:
            logger.info("  超时帧数 : %d（> %d ms 已跳过）", timeout_count, timeout)
        logger.info("-" * 60)
        if arr.mean() <= 60.0:
            logger.info("结论：平均 %.1f ms ≤ 60 ms，满足实时要求 ✓", arr.mean())
        else:
            logger.warning("结论：平均 %.1f ms > 60 ms，存在堆积风险 ✗（建议增大 stride 或减小模型输入）",
                           arr.mean())


if __name__ == "__main__":
    main()
