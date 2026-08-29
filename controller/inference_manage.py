# -*- coding: utf-8 -*-
"""
实时推理管理（controller/inference_manage）
=============================================
职责：实时分析主线程 —— 消费 CameraManage 的最新帧，逐帧检测/姿态 → 状态机切分
      → 逐投篮评分 → 每 REALTIME_STATUS_INTERVAL 秒评估状态提示（print 终端）。

对外暴露：InferenceManage（继承 ThreadBase，分析跑在独立子线程）
依赖：numpy / config / vision_algorithm.segmentation / vision_algorithm.scoring /
      vision_algorithm.llm
"""

import logging
import os
import time

import numpy as np

from config import Config
from common.thread_base import ThreadBase
from common.exceptions import ScoringError
from vision_algorithm.segmentation.shot_segmenter import (
    ShotSegmenter, format_shot_time, format_duration)
from vision_algorithm.scoring.report import (
    score_one_shot, html_to_text, build_report_text)
from vision_algorithm.llm.llm_coach import LLMCoach

logger = logging.getLogger("basketball_scoring")

# 实时逐投评语默认本地规则（offline=True），避免断网每投卡 30s 拖慢实时识别；
# 设 LQ_RT_LLM_ONLINE=1 才实时调用豆包。
_RT_LLM_ONLINE = bool(int(os.environ.get("LQ_RT_LLM_ONLINE", "0")))


class InferenceManage(ThreadBase):
    """实时分析主线程：采样 → 切分 → 逐投篮评分。"""

    def __init__(self, analyzer, std_cache, std_video_count, camera):
        super().__init__(name="InferenceManage")
        self.analyzer = analyzer
        self.std_cache = std_cache
        self.std_video_count = std_video_count
        self.camera = camera
        self.segmenter = None
        self.shot_count = 0
        self.last_person_ts = 0.0
        self.last_ball_ts = 0.0
        self.last_action_ts = 0.0
        self.last_status_ts = 0.0
        self.results = []       # 本次运动识别出的投篮结果（供 /result 查询）
        self.status_msg = ""    # 当前状态提示文案（供 /status 查询）

        # ── 供 HTTP /frames/raw 下发的「最新干净帧 + AI 识别元数据」──
        # latest_preview_frame：预处理后的干净帧（无骨架叠加，544x960）
        # latest_frame_metrics：对应帧的关键点/置信度/角度/框
        self.latest_preview_frame = None
        self.latest_frame_metrics = None

        # 识别是否激活：仅 reset_session() 时置 True（由 /start 触发），
        # /stop 置 False，但推理线程仍持续做帧提取 + 更新 latest_preview_frame，
        # 以保证 Qt 客户端「打开摄像头」即可看到预览画面，无需先「开始运动」。
        self.recognition_active = False

    def reset_session(self):
        """每个运动会话独立：重建切分状态机 + 清零计数/时间戳/结果 + 启用识别。"""
        self.segmenter = ShotSegmenter()
        self.shot_count = 0
        self.last_person_ts = 0.0
        self.last_ball_ts = 0.0
        self.last_action_ts = 0.0
        self.last_status_ts = 0.0
        self.results = []
        self.status_msg = ""
        self.recognition_active = True

    def set_recognition_active(self, active):
        """开启/关闭识别（不停止推理线程，仍持续提供预览帧给 Qt 客户端）。"""
        self.recognition_active = bool(active)

    # ------------------------------------------------------------------
    # 分析主循环
    # ------------------------------------------------------------------
    def _run(self):
        stride = max(1, int(Config.FRAME_STRIDE))
        last_fed_idx = -1
        interval = max(1, Config.REALTIME_STATUS_INTERVAL)
        logger.info("实时分析线程启动（采样步长=%d，状态日志间隔=%ds）", stride, interval)

        while not self.is_stopped():
            frame = self.camera.latest()
            fidx = self.camera.latest_idx
            if frame is None or fidx == last_fed_idx:
                time.sleep(0.005)
                continue
            last_fed_idx = fidx

            # 隔帧采样（与离线评分口径一致）
            if fidx % stride != 0:
                continue

            # 统一用原始大图做检测：竖幅裁剪 + 长边 640 描黑边 + 检测 + player 裁剪姿态，
            # 全程在 _extract_frame_metrics 内完成（描黑边等比缩放，非 RGA 拉伸）。
            # RGA 缩放图不再用于检测（其非等比拉伸不符合「长边 640 + 黑边」要求）。
            det_frame = frame
            frame_orig = None  # 预览直接复用工作帧(544x960)，无需再反算到 1920x1080

            now = time.time()
            try:
                fd = self.analyzer._extract_frame_metrics(
                    det_frame, fidx, frame_orig=frame_orig, ts=now)
            except Exception as e:
                # 推理异常（输入异常/NPU 资源异常）不中断分析线程，记日志后跳过本帧
                logger.error("帧 %d 特征提取失败（%s: %s），跳过本帧",
                             fidx, type(e).__name__, e)
                continue

            # 保存最新干净帧 + 元数据（供 /frames/raw 下发）
            self.latest_frame_metrics = fd
            self.latest_preview_frame = self.analyzer.last_preview_frame

            if fd.get('player_box') is not None:
                self.last_person_ts = now
            if fd.get('ball_boxes'):
                self.last_ball_ts = now

            # 仅在识别激活时（/start 后）喂给 segmenter；打开摄像头后即使未开始运动，
            # 仍持续做帧提取 + 更新 latest_preview_frame，保证 Qt 端能立刻看到预览。
            if self.recognition_active and self.segmenter is not None:
                seg = self.segmenter.feed(fd)
                if seg is not None:
                    self._on_shot(seg, now)

            self._maybe_log_status(now)

    # ------------------------------------------------------------------
    # 逐投篮处理
    # ------------------------------------------------------------------
    def _on_shot(self, seg, now):
        if len(seg['frame_metrics']) < Config.MIN_SHOT_FRAMES:
            logger.warning("实时丢弃过短段: 起点=%d, 出手=%d, 段长=%d < %d",
                           seg['start_idx'], seg['release_idx'],
                           len(seg['frame_metrics']), Config.MIN_SHOT_FRAMES)
            return
        if not seg.get('has_squat'):
            logger.warning("实时丢弃无真实下蹲误检段: 起点=%d, 出手=%d, 段长=%d",
                           seg['start_idx'], seg['release_idx'],
                           len(seg['frame_metrics']))
            return
        self.shot_count += 1
        self.last_action_ts = now
        try:
            shot = self._score_segment(seg)
            self.results.append(shot)
            logger.info(
                "实时识别到第 %d 次投篮：综合 %.1f / 100（起点=%d 出手=%d，"
                "开始时间 %s 结束时间 %s 用时 %s）",
                seg['shot_idx'], shot['scores']['final_score'],
                seg['start_idx'], seg['release_idx'],
                format_shot_time(seg.get('start_time')),
                format_shot_time(seg.get('end_time')),
                format_duration(seg.get('duration')))
        except Exception as e:
            # 打分异常（数组长度不一致/输入数据异常）不中断分析线程，记录分类日志
            err = ScoringError(f"实时评分失败: {e}", kind="scoring", cause=e)
            logger.error("%s", err)

    def _score_segment(self, seg):
        """对实时切出的一段投篮做评分，写 JSON + 文本报告，返回 shot_result。"""
        seq1, seq2, rel_height, idx_squat = self.analyzer._split_and_height(
            seg['frame_metrics'])
        video_fps = getattr(self.analyzer, 'current_fps', 30.0)
        scores, reports = score_one_shot(
            self.std_cache['champ1'], self.std_cache['champ2'],
            self.std_cache['avg_std_height'],
            np.array(seq1), np.array(seq2), rel_height, seg['frame_metrics'],
            video_fps=video_fps)

        coach = LLMCoach(offline=not _RT_LLM_ONLINE)
        ai_report = coach.generate_report(
            scores["stage1_dtw"], scores["stage2_dtw"],
            scores["completeness"], scores["coordination"],
            scores["knee_power"], scores["release_angle"])

        shot_result = {
            "test_video": "(实时流)",
            "output_dir": Config.OUTPUT_DIR,
            "standard_video_count": self.std_video_count,
            "shot_idx": seg['shot_idx'],
            "start_idx": seg['start_idx'],
            "release_idx": seg['release_idx'],
            "idx_squat": idx_squat,
            "start_time": seg.get('start_time'),
            "end_time": seg.get('end_time'),
            "duration": seg.get('duration'),
            "start_time_str": format_shot_time(seg.get('start_time')),
            "end_time_str": format_shot_time(seg.get('end_time')),
            "duration_str": format_duration(seg.get('duration')),
            "clip1_video": None,
            "clip2_video": None,
            "frames_dir": None,
            "scores": scores,
            "reports": {k: html_to_text(v[1]) for k, v in reports.items()},
            "ai_report": ai_report,
        }

        shots_dir = os.path.join(Config.OUTPUT_DIR, "shots")
        os.makedirs(shots_dir, exist_ok=True)
        json_path = os.path.join(shots_dir, f"result_shot{seg['shot_idx']:04d}.json")
        report_path = os.path.join(shots_dir, f"report_shot{seg['shot_idx']:04d}.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(shot_result, f, ensure_ascii=False, indent=2)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(build_report_text(shot_result))
        return shot_result

    # ------------------------------------------------------------------
    # 状态提示（定时评估，print 到终端 + 写入 status_msg 供 /status 查询）
    # ------------------------------------------------------------------
    def _maybe_log_status(self, now):
        interval = max(1, Config.REALTIME_STATUS_INTERVAL)
        if now - self.last_status_ts < interval:
            return
        self.last_status_ts = now
        no_person = (now - self.last_person_ts) > interval
        no_ball = (now - self.last_ball_ts) > interval
        no_action = (now - self.last_action_ts) > interval
        # 三态优先级：无人 → 已投过但动作停 → 有球开始前无球
        if no_person:
            msg = "识别不到投篮者"
        elif self.shot_count > 0 and no_action:
            msg = "请继续投篮"
        elif no_ball:
            msg = "请开始投篮"
        else:
            msg = ""
        self.status_msg = msg
        if msg:
            # 状态提示只打印到终端（stdout），不写日志文件
            print(f"[状态] {msg}", flush=True)
