# -*- coding: utf-8 -*-
"""
摄像头管理（controller/camera_manage）
========================================
职责：RTSP 拉流会话（实时流必须走 RK3588 MPP 硬解，失败直接报错不静默
      回退；USE_MPP_DECODE=false 才走 cv2 软解兜底），维护最新帧；
      录制状态时在拉流线程内同步写录制帧（与录制同线程，保证不丢帧）。

预览说明：
  - 本模块不再用 cv2.imshow 做终端预览（Qt 客户端已接管渲染）。
  - RK3588 只负责「MPP 硬解出 BGR 帧 → 送入 AI 识别评分」，并把干净帧
    通过 HTTP 下发给 Windows Qt 客户端，Qt 端用 QPainter 重建显示。

异常捕获（对应「RTSP 读取异常」需求）：
  - 摄像头断流 / 视频损坏 / 解码失败 统一分类为 RtspStreamError
  - 断流后做有界重连（RTSP_RECONNECT_ATTEMPTS 次），失败置 error 状态，
    并把 last_error 暴露给 /status，供 Qt 客户端通道状态展示
  - 维护 decode_mode / decode_fps / 缓存长度 等通道状态

队列释放（对应「关闭后重开队列满」需求）：
  - close() 显式 join 拉流线程，等待 MPP 解码器 stop + 清空解码队列后返回，
    确保每次重新打开前旧的解码队列已被彻底释放，避免积压阻塞。

对外暴露：CameraManage（继承 ThreadBase，拉流跑在独立子线程）
依赖：cv2 / config / controller.recording_manage / common.exceptions
"""

import logging
import os
import threading
import time
from collections import deque

import cv2

from config import Config
from common.thread_base import ThreadBase
from common.exceptions import RtspStreamError
from controller.recording_manage import RecordingManage
from vision_algorithm.mpp.mpp_config import (
    USE_MPP_DECODE, MPP_DISPLAY_W, MPP_DISPLAY_H)

logger = logging.getLogger("basketball_scoring")

# 断流重连参数（次数/间隔），避免摄像头短暂断流导致进程崩溃
RTSP_RECONNECT_ATTEMPTS = 5
RTSP_RECONNECT_DELAY_SEC = 2.0

# open() 时等待上一次会话拉流线程退出的超时（秒），避免 MPP 解码器清理过慢卡死 HTTP 线程
OPEN_JOIN_TIMEOUT_SEC = 5.0


class CameraManage(ThreadBase):
    """摄像头拉流会话（MPP 硬解优先，失败回退 cv2 软解）。"""

    def __init__(self, analyzer, recording=None):
        super().__init__(name="CameraManage")
        self.analyzer = analyzer
        self.recording = recording or RecordingManage()
        self.lock = threading.RLock()
        self.state = "closed"          # closed / opened / recording / paused / stopped / running / error
        self._latest_frame = None      # 原始大图（预览/录制）
        self._latest_scale = None      # RGA 缩放图（640x640，供检测）
        self.latest_idx = 0
        self._recent_frames = deque(maxlen=10)  # 最近 N 帧缓存（供 /frames 取多帧）

        # ── 通道状态（供 /status + Qt 客户端展示）──
        self.decode_mode = "none"      # none / mpp_hard / cv2_soft
        self.last_error = ""           # 最近一次拉流错误（空=无）
        self._fps_window = deque(maxlen=30)  # 最近帧的时间戳窗口，用于统计解码帧率

    # ------------------------------------------------------------------
    # 拉流线程（ThreadBase._run）
    # ------------------------------------------------------------------
    def _run(self):
        """拉流主循环：硬解优先，断流有界重连，不静默回退软解。"""
        attempts = 0
        while not self.is_stopped():
            try:
                # 实时 RTSP 流必须走 MPP 硬解：
                # 摄像头 H265 实时流参考帧不完整，cv2/ffmpeg 软解会出现
                # "Could not find ref with POC" / "CABAC_MAX_BIN" / "cu_qp_delta out of range"
                # 等无法恢复的报错。默认 MPP 失败直接报错，不静默回退软解。
                if not USE_MPP_DECODE:
                    logger.warning("USE_MPP_DECODE=false，实时流走 cv2 软解（H265 风险自担）")
                    self.decode_mode = "cv2_soft"
                    self._run_soft()
                    return
                self.decode_mode = "mpp_hard"
                self._run_mpp()
                return
            except ImportError as e:
                # mpp_player 未编译/未安装：属部署问题，重连无意义，直接退出并置错误
                self.last_error = f"MPP 硬解库未安装/未编译: {e}"
                self._set_error(RtspStreamError(
                    "MPP 硬解库 mpp_player 未编译/未安装，实时流必须硬解",
                    kind="rtsp", cause=e))
                logger.error("MPP 硬解库未编译/未安装（%s）：实时流必须硬解，"
                             "请在 RK3588 上编译 vision_algorithm/mpp/mpp_player 后再启动", e)
                return
            except RtspStreamError as e:
                self.last_error = str(e)
                self._set_error(e)
                attempts += 1
                if attempts >= RTSP_RECONNECT_ATTEMPTS:
                    logger.error("RTSP 断流重连 %d 次仍失败，停止拉流（state=error）：%s",
                                 attempts, e)
                    return
                logger.warning("RTSP 断流/解码失败（%s），%.1fs 后第 %d/%d 次重连...",
                               e, RTSP_RECONNECT_DELAY_SEC,
                               attempts, RTSP_RECONNECT_ATTEMPTS)
                time.sleep(RTSP_RECONNECT_DELAY_SEC)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self._set_error(RtspStreamError(
                    "RTSP 拉流未知异常", kind="rtsp", cause=e))
                logger.error("RTSP 拉流未知异常（%s: %s）", type(e).__name__, e)
                return

    def _run_mpp(self):
        frames = self.analyzer._iter_rtsp_frames(
            Config.CAMERA_RTSP_URL,
            MPP_DISPLAY_W, MPP_DISPLAY_H)
        for frame, scale in frames:
            if self.is_stopped():
                break
            self._on_frame(frame, scale)

    def _run_soft(self):
        cap = cv2.VideoCapture(Config.CAMERA_RTSP_URL)
        if not cap.isOpened():
            raise RtspStreamError(
                "RTSP 软解无法打开（断流或地址错误）", kind="rtsp")
        try:
            while not self.is_stopped():
                ok, frame = cap.read()
                if not ok:
                    # 软解读帧失败：可能是断流或解码失败
                    raise RtspStreamError(
                        "RTSP 软解读帧失败（断流/视频损坏/解码失败）", kind="rtsp")
                self._on_frame(frame)
        finally:
            cap.release()

    def _set_error(self, exc):
        with self.lock:
            if self.state not in ("closed", "stopped"):
                self.state = "error"

    def _on_frame(self, frame, scale=None):
        # 帧完整性校验（解码失败/坏帧直接丢弃并记录，不让坏帧流入推理/录制）
        if frame is None or getattr(frame, "size", 0) == 0:
            logger.warning("RTSP 解码出空帧，已丢弃")
            return
        need_write = False
        with self.lock:
            self.latest_idx += 1
            self._latest_frame = frame
            if scale is not None and getattr(scale, "size", 0) > 0:
                self._latest_scale = scale
            self._recent_frames.append(frame.copy())
            self._fps_window.append(time.time())
            need_write = (self.recording.is_recording
                          and self.state in ("recording", "running"))
        # 录像写帧移出锁外：避免编码/拷贝阻塞推理线程取帧（锁竞争导致画面卡住）
        if need_write:
            self.recording.write(frame)
            # 每 RECORD_ROTATE_SEC 秒自动分段保存（纯录像与运动识别都生效）
            if time.time() - self.recording.record_rotate_ts >= Config.RECORD_ROTATE_SEC:
                self.recording.rotate()

    # ------------------------------------------------------------------
    # 对外控制
    # ------------------------------------------------------------------
    def open(self):
        """打开摄像头：连接 RTSP 并启动拉流线程（仅预览）。

        重新打开前，先清空上一次会话残留的帧缓存与通道状态，
        避免旧的解码队列/帧缓存影响新一轮拉流。
        """
        with self.lock:
            if self.state != "closed":
                return False, "camera already open"
            os.makedirs(self.recording.record_dir, exist_ok=True)
            # 打开前显式清空残留（防止上次会话未消费完的帧堆积）
            self._latest_frame = None
            self._latest_scale = None
            self._recent_frames.clear()
            self._fps_window.clear()
            self.latest_idx = 0
            self.last_error = ""
            self.decode_mode = "none"
            # 限时等待上一次会话的拉流线程退出（释放 MPP 解码队列），避免阻塞卡死；
            # 旧线程通常数秒内退出，超时则后台继续退出，不阻塞本次打开。
            self.start(join_timeout=OPEN_JOIN_TIMEOUT_SEC)
            self.state = "opened"
        return True, "opened"

    def close(self):
        """关闭摄像头：断开 RTSP，释放资源。

        不阻塞等待拉流线程退出：立即置 closed 并返回（避免 /close 请求超时）。
        旧拉流线程在后台自行退出，其 finally 内 player.stop()+close() 会清空
        MPP 解码队列；下次 open() 时 ThreadBase.start() 会 join 旧线程，确保
        重开前解码队列已被释放。
        """
        with self.lock:
            self.stop()
            self.state = "closed"
            self._latest_frame = None
            self._latest_scale = None
            self._recent_frames.clear()
            self._fps_window.clear()
            self.decode_mode = "none"
        return True, "closed"

    def set_state(self, state):
        """设置会话状态（由协调层 HttpManage 调用）。"""
        with self.lock:
            self.state = state

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def latest(self):
        with self.lock:
            return self._latest_frame

    def latest_scale(self):
        """返回最新 RGA 缩放图（640x640，供检测）。"""
        with self.lock:
            return self._latest_scale

    def recent_frames(self, n=1):
        """返回最近 n 帧（按时间升序）；n<=0 返回空列表。"""
        with self.lock:
            frames = list(self._recent_frames)
        if n <= 0:
            return []
        return frames[-n:]

    def cache_len(self):
        """最近帧缓存长度（供 /status 通道状态展示）。"""
        with self.lock:
            return len(self._recent_frames)

    @property
    def decode_fps(self):
        """估算解码帧率（最近 30 帧的时间窗口内平均）。"""
        with self.lock:
            w = self._fps_window
            if len(w) < 2:
                return 0.0
            span = w[-1] - w[0]
            if span <= 0:
                return 0.0
            return (len(w) - 1) / span

    def status(self):
        with self.lock:
            return {
                "state": self.state,
                "latest_idx": self.latest_idx,
                "decode": self.decode_mode,
                "fps": round(self.decode_fps, 2),
                "cache_len": len(self._recent_frames),
                "last_error": self.last_error,
            }
