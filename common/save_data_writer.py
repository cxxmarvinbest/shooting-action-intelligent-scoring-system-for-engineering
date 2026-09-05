# -*- coding: utf-8 -*-
"""
save_data 异步写盘（common/save_data_writer）
============================================
负责「投篮逐帧图 + data.json」等结构化产物的非阻塞落盘。

设计动机：
  - 主识别链路（inference 线程）每投要写 2×N 张 jpg + 1 个 json，
    若用 cv2.imwrite / json.dump 同步落盘，磁盘抖动会直接拖慢 FSM 喂帧节奏；
  - 录像 / 投篮球员抓帧属实时任务，主线程必须保持「拉流→推理→评分」闭环，
    写盘抖动不能反向阻塞推理；
  - 写盘失败（磁盘满 / IO 错误）只允许打 error 日志，绝不能抛异常打断主流程。

对外暴露：
  - SaveDataWriter：单例风格的异步写盘器（后台线程 + 有界队列 + 背压丢帧）
  - close() / flush()：会话结束时排空队列、释放后台线程

注意：本类不负责「目录创建 / 文件命名」，只负责把内容送到磁盘；
目录与命名由 SaveDataLayout 负责，分层清晰。
"""

import json
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("basketball_scoring")


# 默认队列上限：~200 张 jpg，够覆盖一次连续 5 投 × 40 帧 不会满；
# 满则丢最旧任务（背压），保「实时识别不被写盘拖垮」的优先级。
DEFAULT_QUEUE_SIZE = 256


class SaveDataWriter:
    """异步写盘器：单后台线程 + 有界任务队列 + 失败仅日志。

    任务类型：见 _TaskKind，支持 cv2 image / 任意 json-serializable 对象。
    写盘异常（路径不可写 / 磁盘满 / cv2 编码失败）一律捕获并打 error，
    不抛、不打断主识别线程。
    """

    # 任务类型
    KIND_IMAGE = "image"     # (path:str, img:np.ndarray, jpeg_quality:int)
    KIND_JSON = "json"       # (path:str, data:any)
    KIND_RAW = "raw"         # (path:str, bytes:bytes)

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE,
                 name: str = "SaveDataWriter"):
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = None
        self._name = name
        self._dropped = 0    # 队列满背压丢任务计数（监控用）
        self._failed = 0     # 写盘失败计数（监控用）
        self._done = 0       # 成功落盘计数（监控用）

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """启动后台写盘线程（幂等：已启动时 no-op）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.info("SaveDataWriter 启动：%s（队列上限=%d）",
                    self._name, self._q.maxsize)

    def close(self, timeout: float = 5.0):
        """停止后台线程：排空队列后退出。

        timeout：等待线程退出的最大秒数；超时则放弃剩余任务（只打 warning）。
        会话结束（/stop）必须调本接口，否则后台线程会变成"幽灵线程"持续占资源。
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        if not self._q.empty():
            logger.warning(
                "SaveDataWriter 关闭时仍有 %d 个任务未落盘（timeout=%.1fs）",
                self._q.qsize(), timeout)
            # 排空残留（不写盘，仅清队列，避免继续吃内存）
            self._drain_queue()
        logger.info("SaveDataWriter 关闭：成功=%d, 失败=%d, 背压丢=%d",
                    self._done, self._failed, self._dropped)

    def _drain_queue(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # 对外提交（同步入队，非阻塞；队列满则丢最旧）
    # ------------------------------------------------------------------
    def submit_image(self, path: str, img: np.ndarray,
                     jpeg_quality: int = 90) -> bool:
        """提交一张图（jpg）异步落盘。

        返回 True=成功入队，False=队列满丢弃（已自增 _dropped 计数）。
        """
        if img is None or getattr(img, "size", 0) == 0:
            logger.warning("submit_image 跳过空帧: %s", path)
            return False
        # 转连续数组（防止上游传非连续 BGR）
        img = np.ascontiguousarray(img)
        task = (self.KIND_IMAGE, (path, img, int(jpeg_quality)))
        return self._enqueue(task)

    def submit_json(self, path: str, data) -> bool:
        """提交一段 JSON 异步落盘（ensure_ascii=False，便于中文 key 直读）。"""
        task = (self.KIND_JSON, (path, data))
        return self._enqueue(task)

    def submit_bytes(self, path: str, raw: bytes) -> bool:
        """提交一段原始字节异步落盘（备用，目前未使用）。"""
        if raw is None:
            logger.warning("submit_bytes 跳过空数据: %s", path)
            return False
        task = (self.KIND_RAW, (path, raw))
        return self._enqueue(task)

    def _enqueue(self, task) -> bool:
        """非阻塞入队；满则丢最旧任务（背压）。"""
        try:
            self._q.put_nowait(task)
            return True
        except queue.Full:
            # 背压：丢最旧保最新，避免主线程被同步写盘阻塞
            try:
                self._q.get_nowait()
                self._q.put_nowait(task)
                self._dropped += 1
                return True
            except (queue.Empty, queue.Full):
                self._dropped += 1
                logger.error(
                    "SaveDataWriter 队列背压异常：path=%s 任务已丢弃（累计丢=%d）",
                    task[1][0] if task[1] else "?", self._dropped)
                return False

    # ------------------------------------------------------------------
    # 后台写盘线程
    # ------------------------------------------------------------------
    def _run(self):
        """后台线程主循环：取任务 → 落盘 → 异常仅日志。"""
        while not self._stop.is_set():
            try:
                kind, payload = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._dispatch(kind, payload)
                self._done += 1
            except Exception as e:
                self._failed += 1
                path = payload[0] if payload else "?"
                logger.error(
                    "SaveDataWriter 写盘失败（%s: %s, kind=%s, path=%s）",
                    type(e).__name__, e, kind, path)

    def _dispatch(self, kind: str, payload):
        if kind == self.KIND_IMAGE:
            path, img, jpeg_quality = payload
            # cv2.imencode 失败通常意味着「图像损坏 / 编码器异常」，
            # 单独捕获并打 error，区别于通用 Exception。
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                raise RuntimeError(f"cv2.imencode 失败（path={path}）")
            self._atomic_write_bytes(path, buf.tobytes())
        elif kind == self.KIND_JSON:
            path, data = payload
            text = json.dumps(data, ensure_ascii=False, indent=2)
            self._atomic_write_bytes(path, text.encode("utf-8"))
        elif kind == self.KIND_RAW:
            path, raw = payload
            self._atomic_write_bytes(path, raw)
        else:
            raise ValueError(f"未知任务类型: {kind!r}")

    @staticmethod
    def _atomic_write_bytes(path: str, data: bytes):
        """原子写：先写 .tmp 再 rename，避免进程崩溃导致半截 jpg / json。

        rename 在同分区下是原子操作（POSIX / Windows NTFS 均支持）。
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # 某些文件系统不支持 fsync，吞掉即可
                pass
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # 监控
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """返回写盘统计，供 /status 接口暴露监控指标。"""
        return {
            "queue_size": self._q.qsize(),
            "queue_max": self._q.maxsize,
            "done": self._done,
            "failed": self._failed,
            "dropped": self._dropped,
            "alive": self._thread is not None and self._thread.is_alive(),
        }
