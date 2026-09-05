# -*- coding: utf-8 -*-
"""
实时推理管理（controller/inference_manage）
=============================================
职责：实时分析主线程 —— 消费 CameraManage 的最新帧，逐帧检测/姿态 → 状态机切分
      → 逐投篮评分 → 每 REALTIME_STATUS_INTERVAL 秒评估状态提示（print 终端）。

N1 重构（与「任务重构整理」长文对齐）：
  - 投篮段输出改为 save_data/{date}/{session}/images/00N/
  - 每一投每帧输出 {frame}-src.jpg（原图按 player 框 + margin 裁剪）+ {frame}-ai.jpg
    （同裁剪图叠加 17 点 COCO 骨架）
  - 同步输出 data.json：shot_num / start_time / end_time / start_frame / end_frame /
    joint_meta（17 点语义）/ list_pose（每帧 17 点 x,y,conf）/ scoring（七项分数）
  - 落盘走 SaveDataWriter 异步队列，IO 异常仅打 error 日志，不打断主识别线程
  - 旧路径 data/output/ 不再写入

对外暴露：InferenceManage（继承 ThreadBase，分析跑在独立子线程）
依赖：numpy / config / vision_algorithm.segmentation / vision_algorithm.scoring /
      vision_algorithm.llm / common.save_data_layout / common.save_data_writer
"""

import json
import logging
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import Config
from common.thread_base import ThreadBase
from common.exceptions import ScoringError
from common.profiler import get_realtime_profiler
from common.save_data_layout import SaveDataLayout
from common.save_data_writer import SaveDataWriter
from vision_algorithm.segmentation.shot_segmenter import (
    format_shot_time, format_duration)
from vision_algorithm.segmentation.shot_fsm import ShotFSM
from vision_algorithm.scoring.report import (
    score_one_shot, html_to_text, build_report_text)
from vision_algorithm.llm.llm_coach import LLMCoach

logger = logging.getLogger("basketball_scoring")

# 实时逐投评语默认本地规则（offline=True），避免断网每投卡 30s 拖慢实时识别；
# 设 LQ_RT_LLM_ONLINE=1 才实时调用豆包。
_RT_LLM_ONLINE = bool(int(os.environ.get("LQ_RT_LLM_ONLINE", "0")))

# 投篮逐帧图缓存：最近 N 帧 BGR（供 _on_shot 写 src/ai 裁剪图用）。
# RING_MAX_FRAMES=200 已覆盖 5 分钟窗口；投篮段通常 < 100 帧，够用。
_FRAME_RING_MAXLEN = 256

# 17 点 COCO 关键点语义映射（与 data.json schema 对齐）
JOINT_META = {
    "p0": "鼻子", "p1": "右眼", "p2": "左眼", "p3": "右耳", "p4": "左耳",
    "p5": "右肩", "p6": "左肩", "p7": "右肘", "p8": "左肘", "p9": "右手腕",
    "p10": "左手腕", "p11": "右髋", "p12": "左髋", "p13": "右膝",
    "p14": "左膝", "p15": "右脚踝", "p16": "左脚踝",
}


class InferenceManage(ThreadBase):
    """实时分析主线程：采样 → 切分 → 逐投篮评分 + save_data 落盘。"""

    def __init__(self, analyzer, std_cache, std_video_count, camera,
                 layout: Optional[SaveDataLayout] = None,
                 shot_writer: Optional[SaveDataWriter] = None):
        super().__init__(name="InferenceManage")
        self.analyzer = analyzer
        self.std_cache = std_cache
        self.std_video_count = std_video_count
        self.camera = camera
        self.layout = layout or SaveDataLayout(root=Config.SAVE_DATA_ROOT)
        self.shot_writer = shot_writer or SaveDataWriter(
            queue_size=Config.SAVE_DATA_QUEUE_SIZE)
        self.segmenter = None
        self.shot_count = 0
        self.last_person_ts = 0.0
        self.last_ball_ts = 0.0
        self.last_action_ts = 0.0
        self.last_status_ts = 0.0
        self.results = []       # 本次运动识别出的投篮结果（供 /result 查询）
        self.status_msg = ""    # 当前状态提示文案（供 /status 查询）

        # ── 供 HTTP /frames/raw 下发的「最新干净帧 + AI 识别元数据」──
        self.latest_preview_frame = None
        self.latest_frame_metrics = None

        # ── 投篮逐帧图 BGR 缓存：frame_idx -> 完整预览帧（1280x720）──
        # _on_shot 写盘时按 frame_idx 查 frame，裁剪 player 框 + margin 出 src。
        self._frame_ring: "deque" = deque(maxlen=_FRAME_RING_MAXLEN)

        # ── 识别是否激活：仅 reset_session() 时置 True（由 /start 触发）──
        self.recognition_active = False

        # N2 性能埋点：与 VideoAnalyzer 共享同一个实时链路单例（get_realtime_profiler
        # 模块级缓存）。analyzer 内部打 preprocess/detect/pose/feature，本线程打 fsm/total。
        # PERF_ENABLED=false 时为空埋点器，开销≈0；相关 pf.* 行均可整体注释。
        self._pf = get_realtime_profiler()

    def reset_session(self, session_dir: Optional[str] = None):
        """每个运动会话独立：重建切分状态机 + 清零计数/时间戳/结果 + 启用识别。

        session_dir：当前会话目录（由 http 协调层在 /start 时创建并注入）。
                     会话生命周期以 /start 为创建起点，因此本方法【绝不】兜底建目录——
                     否则会与 http 层重复创建会话目录，造成「会话文件夹翻倍」。
                     未传 session_dir 时直接抛错，暴露调用方遗漏。
        """
        if not session_dir:
            raise ValueError(
                "reset_session 必须传入 session_dir（由 /start 协调层注入），"
                "禁止兜底建目录以避免会话翻倍")
        self.session_dir = session_dir
        self.images_dir = SaveDataLayout.images_dir(session_dir)
        os.makedirs(self.images_dir, exist_ok=True)
        logger.info("InferenceManage 会话目录: %s", self.session_dir)

        self.segmenter = ShotFSM()
        self.shot_count = 0
        self.last_person_ts = 0.0
        self.last_ball_ts = 0.0
        self.last_action_ts = 0.0
        self.last_status_ts = 0.0
        self.results = []
        self.status_msg = ""
        self._frame_ring.clear()
        # 重置跟踪器到无目标状态（新会话重新锁定主球员，避免上一会话残留锁定框）
        self.analyzer.reset_trackers()
        self.recognition_active = True

    def set_recognition_active(self, active):
        """开启/关闭识别（不停止推理线程，仍持续提供预览帧给 Qt 客户端）。"""
        self.recognition_active = bool(active)

    def close_shot_writer(self):
        """会话结束（/stop）调用：排空异步写盘队列、关闭后台线程。

        必须在 http 协调层 /stop 路由末尾调用一次，否则 SaveDataWriter
        后台线程会变成"幽灵线程"持续占资源。
        """
        self.shot_writer.close(timeout=Config.SAVE_DATA_CLOSE_TIMEOUT)

    # ------------------------------------------------------------------
    # 分析主循环
    # ------------------------------------------------------------------
    def _run(self):
        stride = max(1, int(Config.FRAME_STRIDE))
        last_fed_idx = -1
        interval = max(1, Config.REALTIME_STATUS_INTERVAL)
        logger.info("实时分析线程启动（采样步长=%d，状态日志间隔=%ds）", stride, interval)

        # 启动异步写盘线程（幂等）
        self.shot_writer.start()

        while not self.is_stopped():
            # 原子取「同一解码帧」的 (预览帧, RGA 缩放图, 帧号)，避免 frame 与 scale 错位
            frame, scale, fidx = self.camera.latest_pair()
            if frame is None or fidx == last_fed_idx:
                time.sleep(0.005)
                continue
            last_fed_idx = fidx

            # RGA 缩放图(640x360)在解码启动后前几帧可能尚未产出，跳过本帧等待
            if scale is None:
                time.sleep(0.005)
                continue

            # 隔帧采样（与离线评分口径一致）
            if fidx % stride != 0:
                continue

            # 实时 RGA 路径：检测吃 RGA 等比缩放图(640x360)，补黑边后坐标归一化映射回预览(1280x720)，
            # 全程在 _extract_frame_metrics_norm 内完成（省 Python 侧 letterbox 预处理）。
            now = time.time()
            try:
                fd = self.analyzer._extract_frame_metrics_norm(
                    det360=scale, preview=frame, frame_idx=fidx, ts=now)
            except Exception as e:
                # 推理异常（输入异常/NPU 资源异常）不中断分析线程，记日志后跳过本帧
                logger.error("帧 %d 特征提取失败（%s: %s），跳过本帧",
                             fidx, type(e).__name__, e)
                continue

            # 保存最新干净帧 + 元数据（供 /frames/raw 下发）
            self.latest_frame_metrics = fd
            self.latest_preview_frame = self.analyzer.last_preview_frame

            # 投篮逐帧图缓存：按 fidx 存一份完整预览帧（BGR 副本，独立于 self._latest_frame）
            try:
                self._frame_ring.append((fidx, frame.copy()))
            except Exception as e:
                logger.warning("帧 %d 入缓存失败（%s: %s），跳过", fidx,
                               type(e).__name__, e)

            # 回写 AI metrics 给 CameraManage，供 RecordingManage 的 ai 渲染拉取
            try:
                self.camera.set_ai_metrics({
                    "player_box": fd.get("player_box"),
                    "ball_boxes": fd.get("ball_boxes") or [],
                    "kpts": fd.get("kpts"),
                    "side_str": fd.get("side_str"),
                    "angles": fd.get("angles"),
                })
            except Exception as e:
                logger.debug("回写 AI metrics 失败: %s", e)

            if fd.get('player_box') is not None:
                self.last_person_ts = now
            if fd.get('ball_boxes'):
                self.last_ball_ts = now

            # 仅在识别激活时（/start 后）喂给 segmenter；打开摄像头后即使未开始运动，
            # 仍持续做帧提取 + 更新 latest_preview_frame，保证 Qt 端能立刻看到预览。
            seg = None
            if self.recognition_active and self.segmenter is not None:
                self._pf.zone_begin("fsm")          # N2 埋点：FSM feed 区间
                seg_res = self.segmenter.feed(fd)
                self._pf.zone_end("fsm")
                self.shot_count = seg_res.get('shot_count', self.shot_count)
                seg = seg_res.get('shot_event')

            # N2 埋点：单帧链路结束（total）。投篮评分 _on_shot 属事件级重活（DTW 评分 +
            # 教练评语 + 异步落盘），放在 end_frame 之后调用，不计入逐帧 total——
            # 避免「投篮帧」total 出现秒级尖峰，污染 30ms 稳态链路达标判断（N5）。
            self._pf.end_frame()

            if seg is not None:
                self._on_shot(seg, now)

            self._maybe_log_status(now)

    # ------------------------------------------------------------------
    # 投篮逐帧图 + data.json 落盘（N1 重构核心）
    # ------------------------------------------------------------------
    def _save_shot_segment(self, seg, scores):
        """把本次投篮段的每帧裁剪图（src + ai）与 data.json 异步落盘。

        seg   : segmenter 返回的段 dict（含 frame_metrics / start_idx / release_idx / ...）
        scores: score_one_shot 返回的 scores dict
        返回 : (shot_dir, shot_idx)；不抛异常（IO 失败仅日志）。
        """
        try:
            # 1) 自增 shot_idx（基于 images_dir 实际目录推断，避免编号冲突）
            shot_idx = SaveDataLayout.next_shot_idx(self.images_dir)
            shot_dir = SaveDataLayout.shot_dir(self.images_dir, shot_idx)
        except Exception as e:
            logger.error("投篮目录创建失败（%s: %s），跳过本投落盘", type(e).__name__, e)
            return None, None

        # 2) 按帧写入 src + ai 裁剪图
        margin = int(Config.SHOT_CROP_MARGIN)
        jpeg_quality = int(Config.SHOT_JPEG_QUALITY)
        saved = 0
        total = 0
        miss = 0
        for m in seg.get("frame_metrics") or []:
            fidx = m.get("idx")
            if fidx is None:
                continue
            total += 1
            # 从 ring 找原始预览帧
            full = self._lookup_frame(fidx)
            if full is None:
                # 已被 ring 滚动覆盖（异常长段）→ 记录 miss，跳过本帧图但仍写 data.json
                miss += 1
                continue
            try:
                crop, ox, oy = self._crop_player_box(
                    full, m.get("player_box"), margin)
                if crop is None or crop.size == 0:
                    continue
                src_path = SaveDataLayout.shot_frame_path(
                    shot_dir, int(fidx), SaveDataLayout.FRAME_KIND_SRC)
                ai_path = SaveDataLayout.shot_frame_path(
                    shot_dir, int(fidx), SaveDataLayout.FRAME_KIND_AI)
                # src 原图（BGR）
                self.shot_writer.submit_image(src_path, crop, jpeg_quality)
                # ai 渲染图：同裁剪图叠加 17 点 COCO 骨架（kpts 为全图坐标，减裁剪偏移）
                drawn = self._draw_kpts_on_crop(crop, m.get("kpts"), ox, oy)
                self.shot_writer.submit_image(ai_path, drawn, jpeg_quality)
                saved += 1
            except Exception as e:
                # 单帧失败不影响其它帧
                logger.warning("投篮 %d 帧 %d 落盘失败: %s",
                               shot_idx, fidx, e)

        # 3) data.json：按长文 schema 写入
        try:
            data = self._build_shot_data_json(seg, scores, shot_idx)
            json_path = SaveDataLayout.shot_data_json_path(shot_dir)
            self.shot_writer.submit_json(json_path, data)
        except Exception as e:
            logger.error("投篮 %d data.json 组装/落盘失败: %s", shot_idx, e)

        # 诊断日志：total=段内帧数, saved=成功写图帧数, miss=ring 未命中帧数
        if miss > 0:
            logger.warning(
                "投篮 %d 落盘: dir=%s, 段内帧=%d, 写图=%d, ring未命中=%d"
                "（未命中帧将缺 src/ai 图，仅 data.json 完整）",
                shot_idx, shot_dir, total, saved, miss)
        else:
            logger.info("投篮 %d 落盘: dir=%s, 段内帧=%d, 写图=%d",
                        shot_idx, shot_dir, total, saved)
        return shot_dir, shot_idx

    def _build_shot_data_json(self, seg, scores, shot_idx):
        """按「任务重构整理」长文 schema 组装 data.json。

        字段：
          shot_num / start_time / end_time / start_frame / end_frame /
          joint_meta (17 点语义) / list_pose (每帧 kpts) / scoring
        """
        # 起止时间：实时=epoch（format_shot_time 内部判定），离线=相对秒
        st = seg.get("start_time")
        et = seg.get("end_time")
        # 起止帧：start_idx / release_idx（FSM 进入 HOLD 之前的起点 → 下一次 IDLE 之前最后一帧）
        # 按长文："start_frame：FSM 进入 HOLD 的帧；end_frame：下一次进入 IDLE 之前的最后一帧"
        start_frame = seg.get("start_idx")
        end_frame = seg.get("release_idx")
        # start_time / end_time 转 mm:ss.xxx 形式（实时 epoch 自动判别）
        st_str = format_shot_time(st)
        et_str = format_shot_time(et)

        # list_pose：每帧 17 点 [x, y, conf]
        list_pose = []
        if Config.SHOT_JSON_INCLUDE_POSE:
            for m in (seg.get("frame_metrics") or []):
                kpts = m.get("kpts")
                conf = m.get("kpt_conf")
                if kpts is None:
                    continue
                entry = {"frame_id": int(m.get("idx", 0))}
                keypoints = {}
                for i in range(min(17, len(kpts))):
                    kp = kpts[i]
                    if conf is not None and i < len(conf):
                        c = float(conf[i])
                    else:
                        c = 1.0
                    # key 形式 p0/p1/.../p16
                    keypoints[f"p{i}"] = [
                        round(float(kp[0]), 1),
                        round(float(kp[1]), 1),
                        round(c, 4),
                    ]
                entry["keypoints"] = keypoints
                list_pose.append(entry)

        # scoring：长文七项（注意原始 scores key 不完全一致，做映射）
        scoring = {
            "stage1": float(scores.get("stage1_dtw", 0.0)),
            "stage2": float(scores.get("stage2_dtw", 0.0)),
            "completeness": float(scores.get("completeness", 0.0)),
            "coordination": float(scores.get("coordination", 0.0)),
            "knee_power": float(scores.get("knee_power", 0.0)),
            "release_angle": float(scores.get("release_angle", 0.0)),
            "height": float(scores.get("height", 0.0)),
        }

        return {
            "shot_num": int(shot_idx),
            "start_time": st_str,
            "end_time": et_str,
            "start_frame": int(start_frame) if start_frame is not None else None,
            "end_frame": int(end_frame) if end_frame is not None else None,
            "joint_meta": dict(JOINT_META),
            "list_pose": list_pose,
            "scoring": scoring,
        }

    @staticmethod
    def _crop_player_box(frame, player_box, margin):
        """按 player_box + margin 裁剪（clip 到原图边界），返回 (crop, ox, oy)。

        ox / oy 是裁剪起点在原图（预览）坐标系中的偏移，用于把「全图坐标系」的
        关键点平移回裁剪图局部坐标系后再绘制骨架（见 _draw_kpts_on_crop）。

        player_box 为 None 或裁剪退化时返回整帧副本 + 偏移 (0, 0)（兜底，不报错）。
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, 0, 0
        h, w = frame.shape[:2]
        if player_box is None:
            return frame.copy(), 0, 0
        try:
            x1, y1, x2, y2 = player_box
        except Exception:
            return frame.copy(), 0, 0
        x1 = int(max(0, x1 - margin))
        y1 = int(max(0, y1 - margin))
        x2 = int(min(w - 1, x2 + margin))
        y2 = int(min(h - 1, y2 + margin))
        if x1 >= x2 or y1 >= y2:
            return frame.copy(), 0, 0
        return frame[y1:y2, x1:x2].copy(), x1, y1

    @staticmethod
    def _draw_kpts_on_crop(crop, kpts, ox=0, oy=0):
        """在 crop 上叠加 17 点 COCO 骨架 + 关键点。

        kpts 为「全图（预览）坐标系」(17,2)，ox/oy 是 crop 起点在全图中的偏移；
        绘制前先把每个可见点平移回 crop 局部坐标。不可见点(0,0)跳过。
        """
        if crop is None or getattr(crop, "size", 0) == 0 or kpts is None:
            return crop.copy() if crop is not None else None
        drawn = crop.copy()
        try:
            # 平移回 crop 局部坐标；不可见点(0,0)保持 0，避免负偏移后被误画
            kpts_local = []
            for pt in kpts:
                x, y = float(pt[0]), float(pt[1])
                if x > 0 and y > 0:
                    kpts_local.append((x - ox, y - oy))
                else:
                    kpts_local.append((0.0, 0.0))

            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 6),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            ]
            for a, b in skeleton:
                pa, pb = kpts_local[a], kpts_local[b]
                if pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
                    cv2.line(drawn,
                             (int(pa[0]), int(pa[1])),
                             (int(pb[0]), int(pb[1])),
                             (0, 255, 0), 2)
            for i, pt in enumerate(kpts_local):
                if pt[0] > 0 and pt[1] > 0:
                    color = (0, 255, 255) if i <= 4 else (50, 255, 50)
                    cv2.circle(drawn, (int(pt[0]), int(pt[1])), 3, color, -1)
        except Exception:
            return crop.copy()
        return drawn

    def _lookup_frame(self, fidx):
        """在 _frame_ring 中按 fidx 查找完整预览帧（精确匹配）。

        ring 是有界 deque，命中即返回（注意 ring 已滚动覆盖时返回 None）。
        """
        for idx, fr in self._frame_ring:
            if idx == fidx:
                return fr
        return None

    # ------------------------------------------------------------------
    # 逐投篮处理
    # ------------------------------------------------------------------
    def _on_shot(self, seg, now):
        if len(seg['frame_metrics']) < Config.MIN_SHOT_FRAMES:
            logger.warning(
                "实时丢弃过短段: shot_idx=%d, 起点=%d, 出手=%d, 段长=%d < %d, "
                "has_squat=%s, duration=%.3fs, start_time=%s",
                seg.get('shot_idx'), seg['start_idx'], seg['release_idx'],
                len(seg['frame_metrics']), Config.MIN_SHOT_FRAMES,
                seg.get('has_squat'),
                float(seg.get('duration') or 0.0),
                format_shot_time(seg.get('start_time')))
            return
        if not seg.get('has_squat'):
            logger.info("第 %d 投未检测到充分下蹲（屈膝发力评分将偏低）: 起点=%d, 出手=%d, 段长=%d",
                        seg.get('shot_idx'), seg['start_idx'], seg['release_idx'],
                        len(seg['frame_metrics']))
        self.last_action_ts = now
        try:
            shot = self._score_segment(seg)
            self.results.append(shot)
            # N1：投篮逐帧图 + data.json 异步落盘
            shot_dir, shot_idx = self._save_shot_segment(
                seg, shot["scores"])
            if shot_dir is not None:
                # 补一个 session-relative 路径给前端（与 save_data 根相对）
                shot["save_data"] = {
                    "session_dir": self.session_dir,
                    "images_dir": self.images_dir,
                    "shot_dir": shot_dir,
                    "shot_idx": shot_idx,
                }
            logger.info(
                "实时识别到第 %d 次投篮：综合 %.1f / 100（起点=%d 出手=%d，"
                "开始时间 %s 结束时间 %s 用时 %s，落盘=%s）",
                seg['shot_idx'], shot['scores']['final_score'],
                seg['start_idx'], seg['release_idx'],
                format_shot_time(seg.get('start_time')),
                format_shot_time(seg.get('end_time')),
                format_duration(seg.get('duration')),
                shot_dir or "<失败>")
        except Exception as e:
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
            "output_dir": Config.SAVE_DATA_ROOT,   # N1 兼容字段
            "standard_video_count": self.std_video_count,
            "shot_idx": seg['shot_idx'],
            "start_idx": seg['start_idx'],
            "release_idx": seg['release_idx'],
            "idx_squat": idx_squat,
            "has_squat": seg.get('has_squat', False),
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
            print(f"[状态] {msg}", flush=True)
