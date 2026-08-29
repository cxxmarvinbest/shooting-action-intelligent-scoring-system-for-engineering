# -*- coding: utf-8 -*-
"""
录制管理（controller/recording_manage）
==========================================
职责：实时录制器的打开 / 写帧 / rotate / 关闭。

异步写帧设计（解决「开始运动/录像时画面卡住」）：
  - 拉流线程调 write() 只把帧 copy 进有界队列（非阻塞），不再同步做 mp4v 编码；
  - 独立写帧线程从队列取帧做编码写盘，编码耗时不再拖慢拉流/推理线程；
  - 队列满则丢最旧帧（背压），避免堆积。

对外暴露：RecordingManage
依赖：cv2 / config
"""

import logging
import os
import queue
import threading
import time

import cv2

from config import Config

logger = logging.getLogger("basketball_scoring")

# 写帧队列上限（30fps 约 1 秒的缓冲；满则丢最旧帧）
WRITE_QUEUE_SIZE = 30


class RecordingManage:
    """录制器管理（拉流线程异步入队，独立写帧线程编码落盘）。"""

    def __init__(self, record_dir=None):
        self.record_dir = record_dir or Config.RECORD_DIR
        self.recorder = None
        self.current_path = None
        self.record_rotate_ts = 0.0
        self._lock = threading.RLock()
        self._queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self._writer_thread = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # 录制器生命周期
    # ------------------------------------------------------------------
    def open(self):
        """打开录制器，返回 (writer, path)，并启动写帧线程。"""
        with self._lock:
            os.makedirs(self.record_dir, exist_ok=True)
            path = os.path.join(self.record_dir,
                                time.strftime("rec_%Y%m%d_%H%M%S.mp4"))
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            if not writer.isOpened():
                logger.error("录制器打开失败: %s", path)
                return None, None
            self.recorder = writer
            self.current_path = path
            self.record_rotate_ts = time.time()
            self._stop.clear()
            self._writer_thread = threading.Thread(
                target=self._write_loop, name="RecordingWriter", daemon=True)
            self._writer_thread.start()
            logger.info("开始录制: %s", path)
            return writer, path

    def write(self, frame):
        """写一帧（拉流线程调用）：非阻塞入队，队列满则丢最旧帧。"""
        if frame is None:
            return
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()  # 丢最旧，保证入队新帧
                self._queue.put_nowait(frame.copy())
            except Exception:
                pass

    def _write_loop(self):
        """写帧线程：从队列取帧编码写盘（mp4v 编码耗时在此线程，不阻塞拉流/推理）。"""
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                rec = self.recorder
            if rec is not None:
                try:
                    rec.write(frame)
                except Exception:
                    pass

    def _drain_queue(self, rec):
        """把队列中剩余帧写入 rec（rotate/close 前调用）。"""
        while not self._queue.empty():
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                break
            if rec is not None:
                try:
                    rec.write(frame)
                except Exception:
                    pass

    def rotate(self):
        """写满一段自动保存并开新文件（每 RECORD_ROTATE_SEC 秒）。"""
        # 先停写帧线程（锁外，避免与写帧线程抢锁导致 join 超时）
        self._stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3)
        with self._lock:
            rec = self.recorder
            self._drain_queue(rec)
            if rec is not None:
                try:
                    rec.release()
                except Exception:
                    pass
                logger.info("已自动保存一段视频: %s（%d 秒 rotate）",
                            self.current_path, Config.RECORD_ROTATE_SEC)
            self.recorder = None
            self.current_path = None
        # 开新文件（open 内部会重启写帧线程）
        self.open()

    def close(self):
        """关闭录制器，剩余视频完全保存。"""
        # 先停写帧线程（锁外）
        self._stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3)
        with self._lock:
            rec = self.recorder
            self._drain_queue(rec)
            if rec is not None:
                try:
                    rec.release()
                except Exception:
                    pass
                logger.info("录制结束，视频已完全保存: %s", self.current_path)
            self.recorder = None
            self.current_path = None
            # 清空残留（拉流线程可能在 close 瞬间仍塞帧）
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def is_recording(self):
        return self.recorder is not None

    @property
    def path(self):
        return self.current_path
