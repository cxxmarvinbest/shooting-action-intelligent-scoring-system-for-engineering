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
    list_pose（每帧 17 点 x,y,conf，小图坐标）/ scoring（七项分数 + ai_comment）
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
# 大小由 Config.SAVE_FRAME_RING_MAX 决定（默认 128，见 config/save_data.yaml）。
# 只需覆盖「最长投篮段 + CLIP_IMAGE_PREFIX_FRAMES 前扩帧 + 处理滞后余量」；
# 每帧为完整预览图（1280×720×3 ≈ 2.64MB），128 帧 ≈ 338MB，
# 旧值 256 ≈ 675MB 对 RK3588 内存是风险，故下调并可配置。
_FRAME_RING_FALLBACK = 128

# 17 点 COCO 关键点语义映射（与 data.json schema 对齐）
JOINT_META = {
    "p0": "鼻子", "p1": "右眼", "p2": "左眼", "p3": "右耳", "p4": "左耳",
    "p5": "右肩", "p6": "左肩", "p7": "右肘", "p8": "左肘", "p9": "右手腕",
    "p10": "左手腕", "p11": "右髋", "p12": "左髋", "p13": "右膝",
    "p14": "左膝", "p15": "右脚踝", "p16": "左脚踝",
}


# ── N8：AI 骨架图配色规则（BGR）─────────────────────────────────────────
# 分组要求：头部组同色；左右对称关节/连线同色；头/肩/大臂/小臂/躯干/大腿/小腿互相区分。
GROUP_COLORS = {
    "head":      (0, 255, 255),   # 浅黄：鼻 + 双眼 + 双耳
    "shoulder":  (0, 200, 255),   # 橙黄：左右肩
    "upper_arm": (0, 255, 0),     # 绿：左右大臂（肩-肘）
    "forearm":   (0, 140, 255),   # 橙：左右小臂（肘-腕）
    "torso":     (255, 0, 0),     # 蓝：躯干（肩-髋 / 髋-髋）
    "thigh":     (0, 0, 255),     # 红：左右大腿（髋-膝）
    "shank":     (255, 0, 255),   # 品红：左右小腿（膝-踝）
}

# 关键点 idx(0~16) → 分组名（左右对称关节同色）
KPT_GROUP = {
    0: "head", 1: "head", 2: "head", 3: "head", 4: "head",
    5: "shoulder", 6: "shoulder",
    7: "upper_arm", 8: "upper_arm",
    9: "forearm", 10: "forearm",
    11: "torso", 12: "torso",
    13: "thigh", 14: "thigh",
    15: "shank", 16: "shank",
}

# 骨架连线 (a, b, group)：group 决定该连线颜色（左右对称连线同色）
SKELETON = [
    (0, 1, "head"), (0, 2, "head"), (1, 3, "head"), (2, 4, "head"),
    (5, 6, "shoulder"),
    (5, 7, "upper_arm"), (6, 8, "upper_arm"),
    (7, 9, "forearm"), (8, 10, "forearm"),
    (5, 11, "torso"), (6, 12, "torso"), (11, 12, "torso"),
    (11, 13, "thigh"), (12, 14, "thigh"),
    (13, 15, "shank"), (14, 16, "shank"),
]


def _draw_text_with_bg(img, text, org, font_scale=0.5, thickness=1,
                       color=(255, 255, 255), bg_alpha=0.55, margin=4):
    """在 img 上叠加「半透明黑底 + 白字」标签。

    org 为文字左下角（与 cv2.putText 的 org 一致）。
    注意：OpenCV Hershey 字体仅支持 ASCII，请勿传入 ° 等 Unicode 字符。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - margin, y - th - margin),
                  (x + tw + margin, y + baseline + margin), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_angle_text(img, name, value, right_x, y, font_scale=0.5, thickness=1):
    """右上角角度标签：`shoulder 45.3°`；缺失(value=None)显示 `shoulder -`。

    right_x 为标签右边界（右对齐）；度符号用小圆绘制（Hershey 不支持 Unicode °）。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    deg_r = 2                          # 度符号小圆半径
    if value is None:
        text = f"{name} -"
        deg_space = 0
    else:
        text = f"{name} {float(value):.1f}"
        deg_space = deg_r * 2 + 2       # 度符号占位
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = right_x - tw - deg_space
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (right_x, y + baseline + 4),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
    if value is not None:
        # 度符号小圆：画在文字右上角（数字末尾上方）
        cx = x + tw + deg_r + 1
        cy = y - th + deg_r
        cv2.circle(img, (cx, cy), deg_r, (255, 255, 255), -1, cv2.LINE_AA)


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

        # ── 最近一次投篮的落盘编号 + 综合分（供 ai 视频左上角渲染「第几投/得分」）──
        # _on_shot 评分落盘成功后更新；_run 透传给 camera.set_ai_metrics。
        self.latest_shot_idx = None
        self.latest_final_score = None

        # ── MQTT 事件推送（未装配时为 None，_publish_shot_done 判空跳过）──
        self.mqtt = None
        # ── 当前会话 user_id（/start 时由 http 协调层注入，投篮事件携带）──
        self.session_user_id = "0000"

        # ── 供 HTTP /frames/raw 下发的「最新干净帧 + AI 识别元数据」──
        self.latest_preview_frame = None
        self.latest_frame_metrics = None

        # ── 投篮逐帧图 BGR 缓存：frame_idx -> (完整预览帧, 帧 metrics) ──
        # _on_shot 写盘时按 frame_idx 查 frame + fd（player_box/kpts/ball_boxes/
        # angles），裁剪 player 框 + margin 出 src；前后扩帧同样依赖这份缓存。
        ring_max = int(Config.get("SAVE_FRAME_RING_MAX", _FRAME_RING_FALLBACK))
        self._frame_ring: "deque" = deque(maxlen=ring_max)

        # ── 向后扩帧 pending：release_idx 之后的「未来帧」尚未采样，待补写 ──
        # 每元素为 dict：shot_dir/shot_idx/seg/scores/ai_report/offsets/
        # metrics_by_fidx/pending_fidxs。补完或会话结束时生成 data.json 后移除。
        self._pending_shots = []

        # ── 最新已采样帧号（供判断扩展帧是否「未来帧」）──
        self._latest_fidx = -1

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
        # 最近一次投篮的落盘编号/综合分归零（新会话从头计）
        self.latest_shot_idx = None
        self.latest_final_score = None
        # 帧号基准归零：让投篮小图文件名 / data.json 的 frame_id 从「开始运动」起算，
        # 而非「摄像头打开」起算。必须放在 _frame_ring.clear() 之前——解码线程异步
        # 每帧 latest_idx += 1（在 self.lock 内），先归零可保证「归零之后才可能进新帧」，
        # 杜绝 clear 与归零之间插进一帧旧 fidx 造成 ring 里混入孤立脏帧。
        self.camera.reset_frame_idx()
        self._frame_ring.clear()
        self._pending_shots.clear()
        self._latest_fidx = -1
        # 重置跟踪器到无目标状态（新会话重新锁定主球员，避免上一会话残留锁定框）
        self.analyzer.reset_trackers()
        self.recognition_active = True
        # 会话切换后确保异步写盘线程存活：/stop 会调 close_shot_writer() 关闭写盘线程，
        # 但推理线程在 /stop 时不停（仅 set_recognition_active(False)），故连续
        # 「开始运动→停止运动」必须在此重建写盘线程，否则第二次会话的投篮图/data.json
        # 入队后无人消费、永久丢失。start() 幂等，无副作用。
        self.shot_writer.start()

    def set_recognition_active(self, active):
        """开启/关闭识别（不停止推理线程，仍持续提供预览帧给 Qt 客户端）。"""
        self.recognition_active = bool(active)

    def reset_shot_count(self):
        """把投篮计数与分析结果清零（/reset 接口调用，供 Qt 端「重置」按钮）。

        清空计数器与 results（供 /result 查询的分析结果），不停止录像，
        便于连续多组投篮重新开始。同时复位切分状态机（回 IDLE）并清零其
        shot_count，使后续投篮从第 1 投重新编号；latest_shot_idx /
        latest_final_score 归零后 AI 视频左上角 shot/score 回到 '-'。
        """
        self.shot_count = 0
        self.results = []
        self.latest_shot_idx = None
        self.latest_final_score = None
        if self.segmenter is not None:
            # reset() 只清 FSM 帧级状态（回 IDLE），不清 shot_count，需单独清零
            self.segmenter.reset()
            self.segmenter.shot_count = 0
        logger.info("投篮计数与分析结果已清零")

    def set_mqtt(self, mqtt):
        """注入 MQTT 客户端（未启用时为 None，_publish_shot_done 判空跳过）。"""
        self.mqtt = mqtt

    def set_session_user_id(self, user_id):
        """注入当前会话 user_id（/start 时由 http 协调层调用），投篮事件携带。"""
        self.session_user_id = user_id or "0000"

    def close_shot_writer(self):
        """会话结束（/stop）调用：排空异步写盘队列、关闭后台线程。

        必须在 http 协调层 /stop 路由末尾调用一次，否则 SaveDataWriter
        后台线程会变成"幽灵线程"持续占资源。
        """
        # 兜底：未补完的向后扩帧 pending 先提交 data.json（缺的扩展帧不再等），
        # 再排空写盘队列并关闭后台线程，避免向已关闭的 writer 提交任务。
        self._finalize_pending_shots()
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

            # 投篮逐帧图缓存：按 fidx 存一份完整预览帧（BGR 副本）+ 帧 metrics（fd），
            # 供前后扩帧时裁剪小图（fd 含 player_box/kpts/ball_boxes/angles）。
            try:
                self._frame_ring.append((fidx, frame.copy(), fd))
                self._latest_fidx = fidx
            except Exception as e:
                logger.warning("帧 %d 入缓存失败（%s: %s），跳过", fidx,
                               type(e).__name__, e)

            # 向后扩帧 pending 补写：本帧可能命中之前某投的待补跟随帧
            try:
                self._process_pending_shots(fidx, frame, fd)
            except Exception as e:
                logger.warning("pending 补写异常（%s: %s），跳过", type(e).__name__, e)

            # 回写 AI metrics 给 CameraManage，供 RecordingManage 的 ai 渲染拉取。
            # 传入 fidx 作为 metrics 的帧号，CameraManage 按帧号缓存，ai 写帧线程
            # 按帧号取出对应 metrics 渲染，保证骨架与画面逐帧对齐。
            try:
                self.camera.set_ai_metrics({
                    "player_box": fd.get("player_box"),
                    "ball_boxes": fd.get("ball_boxes") or [],
                    "kpts": fd.get("kpts_draw", fd.get("kpts")),
                    "side_str": fd.get("side_str"),
                    "angles": fd.get("angles"),
                    "shot_idx": self.latest_shot_idx,
                    "final_score": self.latest_final_score,
                }, idx=fidx)
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
    def _save_shot_segment(self, seg, scores, ai_report=None):
        """把本次投篮段的每帧裁剪图（src + ai）与 data.json 异步落盘。

        前后扩帧：在状态机确定的 [start_idx, release_idx] 基础上，向前多取
        CLIP_IMAGE_PREFIX_FRAMES 帧（预备动作）、向后多取 CLIP_IMAGE_SUFFIX_FRAMES
        帧（跟随动作）。向后扩展里尚未采样的「未来帧」挂到 _pending_shots，
        待后续帧采样到后由 _process_pending_shots 补写，补完再生成 data.json。

        seg      : segmenter 返回的段 dict（含 frame_metrics / start_idx / release_idx / ...）
        scores   : score_one_shot 返回的 scores dict
        ai_report: LLM 教练评语文本（写入 data.json 的 scoring.ai_comment）
        返回 : (shot_dir, shot_idx)；不抛异常（IO 失败仅日志）。
        """
        try:
            # 1) 自增 shot_idx（基于 images_dir 实际目录推断，避免编号冲突）
            shot_idx = SaveDataLayout.next_shot_idx(self.images_dir)
            shot_dir = SaveDataLayout.shot_dir(self.images_dir, shot_idx)
        except Exception as e:
            logger.error("投篮目录创建失败（%s: %s），跳过本投落盘", type(e).__name__, e)
            return None, None

        # 2) 组装完整帧集合（按 fidx 升序）：向前扩展 + 段内 + 向后扩展
        margin = int(Config.SHOT_CROP_MARGIN)
        jpeg_quality = int(Config.SHOT_JPEG_QUALITY)
        jpeg_optimize = bool(Config.get("SHOT_JPEG_OPTIMIZE", False))
        prefix = max(0, int(Config.CLIP_IMAGE_PREFIX_FRAMES))
        suffix = max(0, int(Config.CLIP_IMAGE_SUFFIX_FRAMES))
        stride = max(1, int(Config.FRAME_STRIDE))
        start_idx = seg.get("start_idx")
        release_idx = seg.get("release_idx")

        seg_metrics = {int(m.get("idx")): m
                       for m in (seg.get("frame_metrics") or [])
                       if m.get("idx") is not None}
        fidx_list = []
        if start_idx is not None:
            for i in range(prefix):
                fidx_list.append(int(start_idx) - stride * (i + 1))
        fidx_list.extend(seg_metrics.keys())
        if release_idx is not None:
            for i in range(suffix):
                fidx_list.append(int(release_idx) + stride * (i + 1))
        fidx_list = sorted(set(fidx_list))

        # 3) 逐帧写 src + ai；向前扩展/段内/已采样向后帧立即写，未来帧挂 pending
        offsets = {}            # fidx -> (ox, oy)，供 data.json 把全图关键点换算为小图坐标
        metrics_by_fidx = {}    # fidx -> metric（实际写图成功帧，供 list_pose）
        pending_fidxs = []      # 尚未采样的向后扩展帧
        saved = 0
        miss = 0
        for fidx in fidx_list:
            if fidx < 0:
                continue
            if fidx in seg_metrics:
                metric = seg_metrics[fidx]
                full, _ = self._lookup_frame_with_metrics(fidx)
            else:
                full, metric = self._lookup_frame_with_metrics(fidx)
            if full is None:
                if fidx > self._latest_fidx:
                    # 未来帧：尚未采样，挂 pending 等后续补写
                    pending_fidxs.append(fidx)
                else:
                    # 已采样但 ring 未命中（被滚动覆盖）→ 记录 miss
                    miss += 1
                continue
            if self._write_shot_frame(shot_dir, shot_idx, fidx, metric, full,
                                      margin, jpeg_quality, jpeg_optimize,
                                      offsets, scores):
                saved += 1
                metrics_by_fidx[fidx] = metric

        # 4) data.json：无 pending 立即生成；有 pending 延后到补写完成
        if pending_fidxs:
            self._pending_shots.append({
                "shot_dir": shot_dir,
                "shot_idx": shot_idx,
                "seg": seg,
                "scores": scores,
                "ai_report": ai_report,
                "offsets": offsets,
                "metrics_by_fidx": metrics_by_fidx,
                "pending_fidxs": sorted(pending_fidxs),
            })
            logger.info(
                "投篮 %d 落盘: dir=%s, 写图=%d, ring未命中=%d, 待补跟随帧=%s",
                shot_idx, shot_dir, saved, miss, sorted(pending_fidxs))
        else:
            self._finalize_shot_data_json(shot_dir, seg, scores, shot_idx,
                                          offsets, ai_report, metrics_by_fidx)
            if miss > 0:
                logger.warning(
                    "投篮 %d 落盘: dir=%s, 写图=%d, ring未命中=%d"
                    "（未命中帧将缺 src/ai 图，仅 data.json 完整）",
                    shot_idx, shot_dir, saved, miss)
            else:
                logger.info("投篮 %d 落盘: dir=%s, 写图=%d",
                            shot_idx, shot_dir, saved)
        return shot_dir, shot_idx

    def _write_shot_frame(self, shot_dir, shot_idx, fidx, metric, full,
                          margin, jpeg_quality, jpeg_optimize, offsets, scores):
        """把单帧裁剪图（src + ai）异步落盘，成功返回 True。

        metric 含 player_box / kpts / ball_boxes / angles（全图/预览坐标）；
        full 为该帧完整预览帧。裁剪退化 / player_box 无效 / 异常时返回 False
        （不落图、不记录 offsets，list_pose 同步剔除，保证与 images 一一对应）。
        """
        try:
            crop, ox, oy = self._crop_player_box(
                full, metric.get("player_box"), margin)
            if crop is None or crop.size == 0:
                return False
            offsets[int(fidx)] = (ox, oy)
            src_path = SaveDataLayout.shot_frame_path(
                shot_dir, int(fidx), SaveDataLayout.FRAME_KIND_SRC)
            ai_path = SaveDataLayout.shot_frame_path(
                shot_dir, int(fidx), SaveDataLayout.FRAME_KIND_AI)
            # src 原图（BGR）
            self.shot_writer.submit_image(src_path, crop, jpeg_quality, jpeg_optimize)
            # ai 渲染图：同裁剪图叠加 17 点 COCO 骨架 + 篮球框（减裁剪偏移）
            drawn = self._draw_kpts_on_crop(
                crop, metric.get("kpts"), ox, oy,
                shot_idx, scores.get("final_score"), metric.get("angles"),
                metric.get("ball_boxes"))
            self.shot_writer.submit_image(ai_path, drawn, jpeg_quality, jpeg_optimize)
            return True
        except Exception as e:
            logger.warning("投篮 %d 帧 %d 落盘失败: %s", shot_idx, fidx, e)
            return False

    def _finalize_shot_data_json(self, shot_dir, seg, scores, shot_idx,
                                 offsets, ai_report, metrics_by_fidx):
        """组装并异步落盘 data.json（list_pose 含前后扩展帧）。"""
        try:
            data = self._build_shot_data_json(
                seg, scores, shot_idx, offsets, ai_report, metrics_by_fidx)
            json_path = SaveDataLayout.shot_data_json_path(shot_dir)
            self.shot_writer.submit_json(json_path, data)
        except Exception as e:
            logger.error("投篮 %d data.json 组装/落盘失败: %s", shot_idx, e)

    def _process_pending_shots(self, fidx, frame, fd):
        """向后扩帧 pending 补写：当前帧 fidx 命中某 pending 待补帧则落盘。

        每采到一帧调用一次；某投待补帧全部处理完后生成该投 data.json。
        「能取多少取多少」：补写帧 player_box 无效时跳过，不报错。
        """
        if not self._pending_shots:
            return
        margin = int(Config.SHOT_CROP_MARGIN)
        jpeg_quality = int(Config.SHOT_JPEG_QUALITY)
        jpeg_optimize = bool(Config.get("SHOT_JPEG_OPTIMIZE", False))
        remaining = []
        for task in self._pending_shots:
            pending = task["pending_fidxs"]
            if fidx in pending:
                self._write_shot_frame(
                    task["shot_dir"], task["shot_idx"], fidx, fd, frame,
                    margin, jpeg_quality, jpeg_optimize,
                    task["offsets"], task["scores"])
                # 写图成功（offsets 含 fidx）才纳入 list_pose
                if fidx in task["offsets"]:
                    task["metrics_by_fidx"][fidx] = fd
                pending.remove(fidx)
            if pending:
                remaining.append(task)
            else:
                self._finalize_shot_data_json(
                    task["shot_dir"], task["seg"], task["scores"],
                    task["shot_idx"], task["offsets"], task["ai_report"],
                    task["metrics_by_fidx"])
        self._pending_shots = remaining

    def _finalize_pending_shots(self):
        """会话结束兜底：未补完的 pending 投篮直接生成 data.json 后清空。"""
        for task in self._pending_shots:
            try:
                self._finalize_shot_data_json(
                    task["shot_dir"], task["seg"], task["scores"],
                    task["shot_idx"], task["offsets"], task["ai_report"],
                    task["metrics_by_fidx"])
            except Exception as e:
                logger.warning("pending 投篮 %s data.json 兜底生成失败: %s",
                               task.get("shot_idx"), e)
        self._pending_shots.clear()

    def _build_shot_data_json(self, seg, scores, shot_idx, offsets=None,
                              ai_report=None, metrics_by_fidx=None):
        """按「任务重构整理」长文 schema 组装 data.json。

        字段：
          shot_num / start_time / end_time / start_frame / end_frame /
          list_pose (每帧 kpts，已换算为小图坐标) / scoring(含 ai_comment)

        offsets  : {fidx: (ox, oy)}，crop 左上角偏移，用于全图关键点 → 小图坐标换算；
                   只对「成功保存图片」的帧存在，缺失帧（player_box 无效 / ring 未命中）
                   会从 list_pose 剔除，保证与 images 图片一一对应、坐标口径统一。
        ai_report: LLM 教练评语文本，写入 scoring.ai_comment。
        metrics_by_fidx: {fidx: metric} 完整写图帧集合（含前后扩展帧）；None 时回退
                    seg.frame_metrics。list_pose 与 start/end_time 顺延均基于它。
        """
        # 起止帧：start_idx / release_idx 前后顺延 prefix/suffix 帧（按 FRAME_STRIDE 步进）
        prefix = max(0, int(Config.CLIP_IMAGE_PREFIX_FRAMES))
        suffix = max(0, int(Config.CLIP_IMAGE_SUFFIX_FRAMES))
        stride = max(1, int(Config.FRAME_STRIDE))
        start_frame = seg.get("start_idx")
        end_frame = seg.get("release_idx")
        if start_frame is not None:
            start_frame = max(0, int(start_frame) - prefix * stride)
        if end_frame is not None:
            end_frame = int(end_frame) + suffix * stride

        # 起止时间：实时=epoch（format_shot_time 内部判定），离线=相对秒。
        # 顺延到扩展后首/尾帧的时间戳；扩展帧缺失（未来帧未采到/ring 未命中）则回退段原始时间。
        st = seg.get("start_time")
        et = seg.get("end_time")
        if metrics_by_fidx:
            sorted_fidxs = sorted(metrics_by_fidx.keys())
            if sorted_fidxs:
                first_m = metrics_by_fidx.get(sorted_fidxs[0])
                last_m = metrics_by_fidx.get(sorted_fidxs[-1])
                if first_m is not None and first_m.get("ts") is not None:
                    st = first_m.get("ts")
                if last_m is not None and last_m.get("ts") is not None:
                    et = last_m.get("ts")
        # start_time / end_time 转 mm:ss.xxx 形式（实时 epoch 自动判别）
        st_str = format_shot_time(st)
        et_str = format_shot_time(et)

        # list_pose：每帧 17 点 [x, y, conf]；遍历完整写图帧集合（含扩展帧），
        # 仅保留成功保存图片的帧（fidx 在 offsets 中），跳过的帧同步剔除，
        # 与 images 图片一一对应、坐标口径统一。
        list_pose = []
        if Config.SHOT_JSON_INCLUDE_POSE:
            if metrics_by_fidx:
                src_metrics = metrics_by_fidx
            else:
                src_metrics = {
                    int(m.get("idx", 0)): m
                    for m in (seg.get("frame_metrics") or [])
                }
            for fidx in sorted(src_metrics.keys()):
                m = src_metrics[fidx]
                # 未写 offsets 的帧 = 未保存图片的帧，list_pose 同步剔除
                if fidx not in (offsets or {}):
                    continue
                kpts = m.get("kpts")
                conf = m.get("kpt_conf")
                if kpts is None:
                    continue
                entry = {"frame_id": fidx}
                # 全图关键点 → 小图坐标：减去该帧 crop 左上角偏移 (ox, oy)
                ox, oy = (offsets or {}).get(fidx, (0.0, 0.0))
                keypoints = {}
                for i in range(min(17, len(kpts))):
                    kp = kpts[i]
                    if conf is not None and i < len(conf):
                        c = float(conf[i])
                    else:
                        c = 1.0
                    # key 形式 p0/p1/.../p16
                    # x/y 取整（像素坐标用整数），conf 保留浮点（round 到 4 位小数）
                    keypoints[f"p{i}"] = [
                        int(round(float(kp[0]) - ox)),
                        int(round(float(kp[1]) - oy)),
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
            "ai_comment": ai_report or "",
        }

        return {
            "shot_num": int(shot_idx),
            "start_time": st_str,
            "end_time": et_str,
            "start_frame": int(start_frame) if start_frame is not None else None,
            "end_frame": int(end_frame) if end_frame is not None else None,
            "list_pose": list_pose,
            "scoring": scoring,
        }

    @staticmethod
    def _crop_player_box(frame, player_box, margin):
        """按 player_box + margin 裁剪（clip 到原图边界），返回 (crop, ox, oy)。

        ox / oy 是裁剪起点在原图（预览）坐标系中的偏移，用于把「全图坐标系」的
        关键点平移回裁剪图局部坐标系后再绘制骨架（见 _draw_kpts_on_crop）。

        player_box 为 None 或裁剪退化时返回 (None, 0, 0)，由上层跳过该帧图片保存
        （不落整张预览图，避免 src/ai 图混入 1280x720 全图）。
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, 0, 0
        h, w = frame.shape[:2]
        if player_box is None:
            return None, 0, 0
        try:
            x1, y1, x2, y2 = player_box
        except Exception:
            return None, 0, 0
        x1 = int(max(0, x1 - margin))
        y1 = int(max(0, y1 - margin))
        x2 = int(min(w - 1, x2 + margin))
        y2 = int(min(h - 1, y2 + margin))
        if x1 >= x2 or y1 >= y2:
            return None, 0, 0
        return frame[y1:y2, x1:x2].copy(), x1, y1

    @staticmethod
    def _draw_kpts_on_crop(crop, kpts, ox=0, oy=0,
                           shot_idx=None, final_score=None, angles=None,
                           ball_boxes=None):
        """在 crop 上叠加 17 点 COCO 骨架 + 关键点 + 篮球框 + 文字标注（N8）。

        - LINE_AA 抗锯齿线条/关键点；
        - 配色规则：头/肩/大臂/小臂/躯干/大腿/小腿分组区分，左右对称关节/连线同色；
        - 篮球框：橙色（与 Ball 检测框一致），全图坐标 → crop 局部坐标后绘制；
        - 左下角：shot:X + scoring:xxx（综合分）；右上角：shoulder/elbow/hip/knee
          角度（缺失显示 "-"）。

        kpts 为「全图（预览）坐标系」(17,2)，ox/oy 是 crop 起点在全图中的偏移；
        绘制前先把每个可见点平移回 crop 局部坐标。不可见点(0,0)跳过。
        ball_boxes 同为「全图（预览）坐标系」，绘制时同样减 (ox, oy) 平移到 crop 局部。
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

            # 1) 骨架连线（LINE_AA + 分组配色）
            for a, b, group in SKELETON:
                pa, pb = kpts_local[a], kpts_local[b]
                if pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
                    cv2.line(drawn, (int(pa[0]), int(pa[1])),
                             (int(pb[0]), int(pb[1])),
                             GROUP_COLORS[group], 2, cv2.LINE_AA)

            # 2) 关键点（分组配色 + LINE_AA 描边）
            for i, pt in enumerate(kpts_local):
                if pt[0] > 0 and pt[1] > 0:
                    color = GROUP_COLORS[KPT_GROUP.get(i, "head")]
                    cv2.circle(drawn, (int(pt[0]), int(pt[1])), 3, color, -1,
                               cv2.LINE_AA)

            # 2.5) 篮球框（全图坐标 → crop 局部坐标，clip 到 crop 范围后绘制）
            if ball_boxes:
                ch, cw = drawn.shape[:2]
                for bb in ball_boxes:
                    try:
                        bx1, by1, bx2, by2 = bb
                    except Exception:
                        continue
                    bx1 = int(round(float(bx1) - ox))
                    by1 = int(round(float(by1) - oy))
                    bx2 = int(round(float(bx2) - ox))
                    by2 = int(round(float(by2) - oy))
                    # clip 到 crop 范围，避免越界坐标
                    bx1 = max(0, min(cw - 1, bx1))
                    by1 = max(0, min(ch - 1, by1))
                    bx2 = max(0, min(cw - 1, bx2))
                    by2 = max(0, min(ch - 1, by2))
                    if bx1 >= bx2 or by1 >= by2:
                        continue
                    cv2.rectangle(drawn, (bx1, by1), (bx2, by2),
                                  (0, 165, 255), 2, cv2.LINE_AA)
                    _draw_text_with_bg(drawn, "Ball", (bx1, max(14, by1 - 4)))

            # 3) 左下角：shot:X + scoring:xxx（综合分）
            _draw_text_with_bg(
                drawn, f"shot:{shot_idx if shot_idx is not None else '-'}",
                (6, 22))
            score_txt = (f"scoring: {float(final_score):.1f}"
                         if final_score is not None else "scoring: -")
            _draw_text_with_bg(drawn, score_txt, (6, 44))

            # 4) 右上角：shoulder/elbow/hip/knee 角度（右对齐，缺失 "-"）
            w = drawn.shape[1]
            right_x = w - 6
            for i, name in enumerate(("shoulder", "elbow", "hip", "knee")):
                val = angles[i] if (angles is not None and i < len(angles)) else None
                _draw_angle_text(drawn, name, val, right_x, 20 + i * 22)
        except Exception:
            return crop.copy()
        return drawn

    def _lookup_frame_with_metrics(self, fidx):
        """在 _frame_ring 中按 fidx 查找 (完整预览帧, 帧 metrics)（精确匹配）。

        返回 (frame, fd)；未命中返回 (None, None)。fd 含 player_box/kpts/
        ball_boxes/angles 等，供前后扩帧裁剪 + 叠加骨架使用。
        ring 是有界 deque，命中即返回（注意 ring 已滚动覆盖时返回 None）。
        """
        for idx, fr, fd in self._frame_ring:
            if idx == fidx:
                return fr, fd
        return None, None

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
                seg, shot["scores"], shot.get("ai_report"))
            # 记录最近一次投篮的编号/综合分，供 ai 视频左上角渲染「第几投/得分」
            if shot_idx is not None:
                self.latest_shot_idx = shot_idx
                self.latest_final_score = shot["scores"].get("final_score")
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
            # MQTT 投篮完成事件推送（未装配 MQTT 时零影响）
            self._publish_shot_done(shot, shot_dir, shot_idx)
        except Exception as e:
            err = ScoringError(f"实时评分失败: {e}", kind="scoring", cause=e)
            logger.error("%s", err)

    def _score_segment(self, seg):
        """对实时切出的一段投篮做评分，写 JSON + 文本报告，返回 shot_result。"""
        seq1, seq2, rel_height, idx_squat = self.analyzer._split_and_height(
            seg['frame_metrics'])
        video_fps = getattr(self.analyzer, 'current_fps', 25.0)
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

        # 聚合投篮者 / 篮球平均置信度（供后端上传 player_confidence_avg /
        # basketball_confidence_avg）。player_conf 由检测链路写入帧元数据；
        # ball_confs 为每帧有效篮球置信度列表（filter_balls 逐帧产出）。
        player_confs = [m.get('player_conf') for m in seg['frame_metrics']
                        if m.get('player_conf') is not None]
        ball_confs = [c for m in seg['frame_metrics']
                      for c in (m.get('ball_confs') or [])]
        player_conf_avg = (round(sum(player_confs) / len(player_confs), 4)
                           if player_confs else 0.0)
        ball_conf_avg = (round(sum(ball_confs) / len(ball_confs), 4)
                         if ball_confs else 0.0)

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
            "player_confidence_avg": player_conf_avg,
            "basketball_confidence_avg": ball_conf_avg,
        }
        return shot_result

    def _publish_shot_done(self, shot, shot_dir, shot_idx):
        """投篮完成 → MQTT 推送轻量事件（未装配 MQTT 时零影响）。

        payload 只放 shot_id / user_id / 得分 / 时间 / detail_url，图片视频不进报文；
        完整 JSON 与资源 url 由 APP 拿 shot_id 走 HTTP /result/<shot_id> 拉取（验收②④）。
        """
        if self.mqtt is None:
            return
        try:
            scores = shot.get("scores") or {}
            sid = ("%03d" % int(shot_idx)) if shot_idx is not None else None
            detail_url = None
            if sid is not None:
                base = Config.get("HTTP_PUBLIC_BASE_URL", "") or \
                    "http://%s:%d" % (Config.get("HTTP_HOST", "127.0.0.1"),
                                      int(Config.get("HTTP_PORT", 8899)))
                detail_url = base.rstrip("/") + "/result/" + sid
            summary = {
                "shot_id": sid,
                "user_id": self.session_user_id or "0000",
                "final_score": scores.get("final_score"),
                "scores": scores,
                "start_time": shot.get("start_time_str"),
                "end_time": shot.get("end_time_str"),
                "duration": shot.get("duration_str"),
                "detail_url": detail_url,
            }
            self.mqtt.publish_shot_done(summary)
        except Exception as e:
            logger.warning("投篮 %s MQTT 事件推送失败: %s", shot_idx, e)

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
