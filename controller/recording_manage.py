# -*- coding: utf-8 -*-
"""
录制管理（controller/recording_manage）
==========================================
职责：实时录制器的打开 / 写帧 / rotate / 关闭。

N1 重构（与「任务重构整理」长文对齐）：
  - 视频输出目录由原 data/output/videos 改为 save_data/{date}/{session}/videos
  - 文件命名：01-20260903_150343-150843_{raw|ai}.mp4
    raw: 原始录制视频
    ai:  同步渲染视频（叠加 player 检测框 + 17 点 COCO 骨架），由 ai write thread
         异步从 CameraManage 拉取最新 AI metrics 后渲染写入
  - 5 分钟自动 rotate（RECORD_ROTATE_SEC）；未满 5 分钟 stop 时立刻落盘
  - 分辨率统一 1280x720（由 Config.PREVIEW_WIDTH/HEIGHT 决定，全局唯一来源）

异步写帧设计（解决「开始运动/录像时画面卡住」）：
  - 拉流线程调 write() 只把帧 copy 进有界队列（非阻塞），不再同步做 mp4v 编码；
  - 独立写帧线程从队列取帧做编码写盘，编码耗时不再拖慢拉流/推理线程；
  - 队列满则丢最旧帧（背压），避免堆积。

双 writer（raw + ai）：
  - 同一帧同时入两个独立队列，由两条写帧线程并行编码；
  - ai 写帧线程额外从 CameraManage 拉 AI metrics（player_box / kpts）做轻渲染。

对外暴露：RecordingManage
依赖：cv2 / config / common.save_data_layout
"""

import logging
import os
import queue
import threading
import time

import cv2

from config import Config
from common.save_data_layout import SaveDataLayout
from common.ffmpeg_writer import FFmpegWriter

logger = logging.getLogger("basketball_scoring")

# 写帧队列上限（30fps 约 1 秒的缓冲；满则丢最旧帧）
WRITE_QUEUE_SIZE = 30

# ai 视频渲染后端：CameraManage 暴露 latest_ai_metrics() 返回最新一帧的
# player_box / kpts / side_str；ai 写帧线程按需取一次，与 frame 近似同步。
# 若 metrics 暂无（推理未跟上），ai 写 raw 副本 + 标记"无 AI 元数据"。


def _draw_box(img, x1, y1, x2, y2, color=(255, 255, 255), alpha=0.30):
    """在 img 上叠加一个半透明实心矩形（文字区块背景）。

    color —— BGR 色值；alpha —— 背景不透明度（0~1）。
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _draw_overlay_block(img, rows, font_scale=1.0, thickness=2,
                        bg_color=(255, 255, 255), bg_alpha=0.30,
                        x0=0, y0=0, right_align=False, right_x=0,
                        line_gap=6, margin=6):
    """绘制一个「半透明白底框 + 多行文字」覆盖层区块。

    rows: [(text, has_deg), ...]
      text    —— 已格式化的 ASCII 文本（Hershey 不支持中文/Unicode）
      has_deg —— 文本末尾是否追加一个度符号小圆（右上角角度用）
    - 左对齐：x0/y0 为区块左上角；右对齐：right_x 为区块右边界，文字右对齐。
    - 字号/描边/白底框颜色与透明度由 recording.yaml 的 RECORD_OVERLAY_* 控制。

    一个区块只画一个白底框（覆盖该区块所有行），非逐行各自画框。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 度符号半径随字号缩放（基准 font_scale=0.5 → 半径 2）
    deg_r = max(2, int(round(2 * font_scale / 0.5)))

    measured = []
    for text, has_deg in rows:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        deg_space = (deg_r * 2 + 2) if has_deg else 0
        measured.append((text, has_deg, tw, th, baseline, deg_space))

    line_h = max(m[3] + m[4] for m in measured) + line_gap
    block_w = max(m[2] + m[5] for m in measured)          # 文本宽 + 度符号占位
    block_h = line_h * len(measured) - line_gap

    # 白底框（一个区块一个框）
    if right_align:
        x1 = right_x - block_w - margin
        x2 = right_x + margin
    else:
        x1 = x0 - margin
        x2 = x0 + block_w + margin
    _draw_box(img, x1, y0 - margin, x2, y0 + block_h + margin, bg_color, bg_alpha)

    # 逐行写文字（首行基线 = y0 + th）
    for i, (text, has_deg, tw, th, baseline, deg_space) in enumerate(measured):
        base_y = y0 + th + i * line_h
        tx = (right_x - tw - deg_space) if right_align else x0
        cv2.putText(img, text, (int(tx), int(base_y)), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        if has_deg:
            cx = tx + tw + deg_r + 1
            cy = base_y - th + deg_r
            cv2.circle(img, (int(cx), int(cy)), deg_r, (255, 255, 255), -1, cv2.LINE_AA)


class RecordingManage:
    """录制器管理（拉流线程异步入队，独立双写帧线程编码落盘）。"""

    def __init__(self, session_dir=None, layout: SaveDataLayout = None,
                 camera=None):
        """初始化。

        session_dir：本会话目录（由 http 协调层在 /start 时创建并注入）。
                     【会话生命周期】会话以 /start 为创建起点，因此本构造
                     器【绝不】在 session_dir 为空时兜底建目录——否则
                     pipeline.py 里 CameraManage(analyzer) 构造 RecordingManage()
                     会在程序启动时就多建一个会话目录，导致「启动一次多一个
                     会话文件夹」的翻倍 bug。
        layout：可选 SaveDataLayout 实例；不传则按 Config.SAVE_DATA_ROOT 新建。
        camera：可选 CameraManage 引用，供 ai 视频渲染拉 AI metrics。
        """
        self.layout = layout or SaveDataLayout(root=Config.SAVE_DATA_ROOT)
        # 会话目录必须由 /start 显式注入（set_session_dir），此处不建目录。
        # session_dir=None 表示「尚无活跃会话」，open() 会拒绝启动录像。
        self.session_dir = session_dir
        if session_dir:
            self.videos_dir = SaveDataLayout.videos_dir(session_dir)
            os.makedirs(self.videos_dir, exist_ok=True)
        else:
            self.videos_dir = None

        # 兼容旧代码可能读 record_dir 的兜底
        self.record_dir = self.videos_dir

        # raw + ai 双 writer
        self.raw_writer = None
        self.ai_writer = None
        self.raw_path = None
        self.ai_path = None

        self.record_rotate_ts = 0.0
        self.clip_idx = 0    # 当前分片编号
        self.clip_start_ts = 0.0  # 当前分片起始时间戳

        self._lock = threading.RLock()
        self._raw_queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self._ai_queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self._raw_thread = None
        self._ai_thread = None
        self._stop = threading.Event()

        self.camera = camera   # 用于 ai 渲染拉 AI metrics

        # ── 诊断计数：实际写入 VideoWriter 的帧数（排查 0 字节视频用）──
        self._raw_written = 0
        self._ai_written = 0
        self._write_fail = 0

    # ------------------------------------------------------------------
    # 录制器生命周期
    # ------------------------------------------------------------------
    def open(self):
        """打开录制器：建立 raw + ai 双 writer，启动双写帧线程。

        返回 (raw_writer, raw_path, ai_writer, ai_path)。
        若 session_dir 未注入（/start 未先 set_session_dir），直接返回 None，
        拒绝在没有会话目录的情况下启动录像。
        """
        # 会话目录必须已注入（/start 调 set_session_dir），否则无法定位视频输出路径
        if not self.session_dir or not self.videos_dir:
            logger.error("RecordingManage.open() 拒绝启动：session_dir 未注入，"
                         "请先 POST /start 建立会话")
            return None, None, None, None
        os.makedirs(self.videos_dir, exist_ok=True)
        with self._lock:
            self.clip_idx = SaveDataLayout.next_clip_idx(self.videos_dir)
            now = time.time()
            self.clip_start_ts = now
            self.raw_path = SaveDataLayout.build_video_path(
                self.videos_dir, self.clip_idx, now, now,
                SaveDataLayout.VIDEO_KIND_RAW)
            self.ai_path = SaveDataLayout.build_video_path(
                self.videos_dir, self.clip_idx, now, now,
                SaveDataLayout.VIDEO_KIND_AI)

            w = Config.PREVIEW_WIDTH
            h = Config.PREVIEW_HEIGHT
            # N5 改造：改用 FFmpegWriter（subprocess 调 ffmpeg + libx264 软件编码）。
            # 原因：cv2.VideoWriter(fourcc='avc1') 在 RK3588 会命中硬件编码器
            # h264_v4l2m2m 且失败后不 fallback 到 libx264，故改用 subprocess 显式
            # 传 -c:v libx264，绕开硬件编码器并精确控制码率。
            codec = getattr(Config, 'FFMPEG_CODEC', 'libx264')
            bitrate = getattr(Config, 'FFMPEG_BITRATE', '600k')
            maxrate = getattr(Config, 'FFMPEG_MAXRATE', bitrate)
            bufsize = getattr(Config, 'FFMPEG_BUF_SIZE', None)
            preset = getattr(Config, 'FFMPEG_PRESET', 'veryfast')
            threads = getattr(Config, 'FFMPEG_THREADS', 0)
            fps = float(Config.get("CAMERA_FPS", 25.0))

            # raw writer（始终开）
            self.raw_writer = FFmpegWriter(
                self.raw_path, w, h, fps,
                codec=codec, bitrate=bitrate,
                maxrate=maxrate, bufsize=bufsize,
                preset=preset, threads=threads)
            if not self.raw_writer.isOpened():
                logger.error("raw 录制器打开失败: %s", self.raw_path)
                self.raw_writer = None
                return None, None, None, None

            # ai writer（按开关）
            self.ai_writer = None
            if Config.RECORD_AI_CLIP:
                self.ai_writer = FFmpegWriter(
                    self.ai_path, w, h, fps,
                    codec=codec, bitrate=bitrate,
                    maxrate=maxrate, bufsize=bufsize,
                    preset=preset, threads=threads)
                if not self.ai_writer.isOpened():
                    logger.warning("ai 录制器打开失败，回退为不写 ai 视频: %s",
                                   self.ai_path)
                    self.ai_writer = None

            self.record_rotate_ts = now
            self._stop.clear()
            # 每个分片清零写入计数
            self._raw_written = 0
            self._ai_written = 0
            self._write_fail = 0
            logger.info("录制器打开: raw=%s (编码器=%s, 码率=%s, fps=%.1f, 尺寸=%dx%d)",
                        self.raw_path, codec, bitrate, fps, w, h)

            # 启动双写帧线程
            self._raw_thread = threading.Thread(
                target=self._raw_write_loop, name="RawClipWriter", daemon=True)
            self._raw_thread.start()
            if self.ai_writer is not None:
                self._ai_thread = threading.Thread(
                    target=self._ai_write_loop, name="AiClipWriter", daemon=True)
                self._ai_thread.start()

            logger.info("开始录制: raw=%s%s",
                        self.raw_path,
                        f", ai={self.ai_path}" if self.ai_writer else " (ai 已关闭)")
            return self.raw_writer, self.raw_path, self.ai_writer, self.ai_path

    def write(self, frame, frame_idx=None):
        """写一帧（拉流线程调用）：非阻塞入 raw + ai 双队列。

        ai 队列只在 RECORD_AI_CLIP=true 时入队；队列满则丢最旧帧。
        ai 队列存 (frame, frame_idx)：渲染时按 frame_idx 从 CameraManage 取
        对应 metrics，保证骨架与画面逐帧对齐（避免「骨架提前于动作」的时间戳错位）。
        帧尺寸与 writer 不匹配时告警一次（避免静默写坏帧导致 0 字节视频）。
        """
        if frame is None:
            return
        # 尺寸一致性校验（一次性告警，避免每帧刷日志）
        try:
            fh, fw = frame.shape[:2]
            if (fw, fh) != (Config.PREVIEW_WIDTH, Config.PREVIEW_HEIGHT) \
                    and not getattr(self, "_size_warned", False):
                logger.warning(
                    "录制帧尺寸异常: 实际=%dx%d, writer=%dx%d（MPP 输出与预览尺寸不一致，"
                    "可能导致视频写坏/0 字节）",
                    fw, fh, Config.PREVIEW_WIDTH, Config.PREVIEW_HEIGHT)
                self._size_warned = True
        except Exception:
            pass
        try:
            self._raw_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._raw_queue.get_nowait()
                self._raw_queue.put_nowait(frame.copy())
            except Exception:
                pass
        if self.ai_writer is not None:
            try:
                self._ai_queue.put_nowait((frame.copy(), frame_idx))
            except queue.Full:
                try:
                    self._ai_queue.get_nowait()
                    self._ai_queue.put_nowait((frame.copy(), frame_idx))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 写帧线程
    # ------------------------------------------------------------------
    def _raw_write_loop(self):
        """raw 写帧线程：从 raw 队列取帧 → 编码写盘（mp4v 编码耗时在此线程）。"""
        while not self._stop.is_set():
            try:
                frame = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                w = self.raw_writer
            if w is not None:
                try:
                    if w.write(frame):
                        self._raw_written += 1
                    else:
                        self._write_fail += 1
                except Exception as e:
                    self._write_fail += 1
                    if self._write_fail <= 3:  # 只打前 3 次，避免刷屏
                        logger.warning("raw 写帧失败（%s: %s）", type(e).__name__, e)

    def _ai_write_loop(self):
        """ai 写帧线程：取帧 → 按帧号取对应 AI metrics → 渲染骨架/框 → 编码写盘。

        帧与 metrics 通过 frame_idx 绑定：采集线程入队时附带帧号，
        推理线程把 metrics 按帧号存入 CameraManage 缓存，此处按帧号取出
        对应 metrics 渲染，保证骨架与画面逐帧对齐（不出现「骨架提前于动作」）。
        """
        while not self._stop.is_set():
            try:
                frame, frame_idx = self._ai_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                w = self.ai_writer
            if w is None:
                continue
            try:
                # 按帧号取对应 metrics；未命中则不传 metrics（仅写 raw 副本）
                metrics = None
                if self.camera is not None and frame_idx is not None:
                    metrics = self.camera.get_ai_metrics_by_idx(frame_idx)
                drawn = self._render_ai_frame(frame, metrics=metrics)
                if w.write(drawn):
                    self._ai_written += 1
                else:
                    self._write_fail += 1
            except Exception as e:
                self._write_fail += 1
                if self._write_fail <= 3:
                    logger.warning("ai 帧渲染/写盘失败: %s", e)

    def _render_ai_frame(self, frame, metrics=None):
        """把 AI metrics（player_box + kpts + angles + shot/score）叠加到 frame 上。

        - 球员框 / 篮球框 / 17 点 COCO 骨架：原图叠加；
        - 左上角：第几投（shot）+ 得分（score）；
        - 右上角：shoulder/elbow/hip/knee 关节角度。

        metrics 由调用方按帧号从 CameraManage 取出（保证与本帧对齐）；
        若 metrics 为 None（推理未跟上 / 帧号缓存未命中），返回 frame 副本，
        保证 ai 视频不中断（仅该帧无骨架叠加）。
        """
        drawn = frame.copy()
        if metrics is None:
            return drawn

        # 1) player 框
        pb = metrics.get("player_box")
        if pb is not None:
            try:
                x1, y1, x2, y2 = pb
                cv2.rectangle(drawn, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 144, 30), 2)
            except Exception:
                pass

        # 2) ball 框
        for bx in (metrics.get("ball_boxes") or []):
            try:
                x1, y1, x2, y2 = bx
                cv2.rectangle(drawn, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 165, 255), 2)
            except Exception:
                pass

        # 3) 17 点 COCO 骨架
        kpts = metrics.get("kpts")
        if kpts is not None:
            try:
                # 简单骨架连线（COCO 17 点常用配对）
                skeleton = [
                    (0, 1), (0, 2), (1, 3), (2, 4),         # 眼/耳
                    (5, 7), (7, 9), (6, 8), (8, 10),        # 手臂
                    (5, 6),                                  # 肩
                    (5, 11), (6, 12), (11, 12),              # 躯干
                    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
                ]
                for a, b in skeleton:
                    pa, pb_ = kpts[a], kpts[b]
                    if pa[0] > 0 and pb_[0] > 0:
                        cv2.line(drawn,
                                 (int(pa[0]), int(pa[1])),
                                 (int(pb_[0]), int(pb_[1])),
                                 (220, 110, 0), 2)
                for i, pt in enumerate(kpts):
                    if pt[0] > 0:
                        color = (0, 255, 255) if i <= 4 else (50, 255, 50)
                        cv2.circle(drawn, (int(pt[0]), int(pt[1])),
                                   3, color, -1)
            except Exception:
                pass

        # 4) 左上角：第几投（shot 编号，1 起）+ 得分（最近一次投篮综合分）。
        #    label 用 ASCII（Hershey 不支持中文）：shot=第几投，score=得分。
        #    shot_idx / final_score 未产出（尚未完成第一投）时显示 '-'。
        #    字号/描边/白底框透明度与颜色统一由 recording.yaml 的 RECORD_OVERLAY_* 控制。
        try:
            font_scale = float(Config.get("RECORD_OVERLAY_FONT_SCALE", 1.0))
            thickness = int(Config.get("RECORD_OVERLAY_THICKNESS", 2))
            bg_alpha = float(Config.get("RECORD_OVERLAY_BG_ALPHA", 0.30))
            bg_color = tuple(Config.get("RECORD_OVERLAY_BG_COLOR", (255, 255, 255)))

            shot_idx = metrics.get("shot_idx")
            final_score = metrics.get("final_score")
            left_rows = [
                (f"shot:{shot_idx if shot_idx is not None else '-'}", False),
                (f"score: {float(final_score):.1f}"
                 if final_score is not None else "score: -", False),
            ]
            _draw_overlay_block(drawn, left_rows, font_scale, thickness,
                                bg_color, bg_alpha, x0=6, y0=6)

            # 5) 右上角：shoulder/elbow/hip/knee 关节角度（右对齐，缺失 "-"）
            angles = metrics.get("angles")
            right_rows = []
            for i, name in enumerate(("shoulder", "elbow", "hip", "knee")):
                val = angles[i] if (angles is not None and i < len(angles)) else None
                right_rows.append(
                    (f"{name} {float(val):.1f}", True) if val is not None
                    else (f"{name} -", False))
            right_x = drawn.shape[1] - 6
            _draw_overlay_block(drawn, right_rows, font_scale, thickness,
                                bg_color, bg_alpha,
                                right_align=True, right_x=right_x, y0=6)
        except Exception:
            pass

        return drawn

    def _drain_queue(self, q, writer):
        """把队列中剩余帧写入 writer，返回实际写入帧数。

        raw 队列元素是 ndarray 帧；ai 队列元素是 (frame, frame_idx) 元组，
        此处取出 frame 部分直接写（rotate/close 时不再做渲染对齐，仅落盘）。
        """
        n = 0
        while not q.empty():
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            # ai 队列元素为 (frame, frame_idx) 元组，raw 队列为 ndarray
            if isinstance(item, tuple):
                frame = item[0]
            else:
                frame = item
            if writer is not None and frame is not None:
                try:
                    writer.write(frame)
                    n += 1
                except Exception:
                    pass
        return n

    @staticmethod
    def _file_size(path):
        """返回文件字节数；文件不存在返回 -1。"""
        try:
            return os.path.getsize(path) if path and os.path.exists(path) else -1
        except OSError:
            return -1

    # ------------------------------------------------------------------
    # rotate / close
    # ------------------------------------------------------------------
    def rotate(self):
        """写满一段自动保存并开新文件（每 RECORD_ROTATE_SEC 秒）。"""
        # 先停写帧线程（锁外，避免与写帧线程抢锁导致 join 超时）
        self._stop.set()
        if self._raw_thread is not None:
            self._raw_thread.join(timeout=3)
        if self._ai_thread is not None:
            self._ai_thread.join(timeout=3)
        with self._lock:
            self._drain_queue(self._raw_queue, self.raw_writer)
            self._drain_queue(self._ai_queue, self.ai_writer)
            if self.raw_writer is not None:
                try:
                    self.raw_writer.release()
                except Exception:
                    pass
                sz = self._file_size(self.raw_path)
                logger.info("已自动保存一段视频: %s（%d 秒 rotate）写入帧数=%d, 文件大小=%d 字节%s",
                            self.raw_path, Config.RECORD_ROTATE_SEC,
                            self._raw_written, sz,
                            " [警告: 0 字节，编码器可能不可用]" if sz == 0 else "")
            if self.ai_writer is not None:
                try:
                    self.ai_writer.release()
                except Exception:
                    pass
                sz = self._file_size(self.ai_path)
                logger.info("已自动保存一段 ai 视频: %s 写入帧数=%d, 文件大小=%d 字节%s",
                            self.ai_path, self._ai_written, sz,
                            " [警告: 0 字节]" if sz == 0 else "")
            self.raw_writer = None
            self.ai_writer = None
            self.raw_path = None
            self.ai_path = None
            self._raw_thread = None
            self._ai_thread = None
        # 开新文件（open 内部会重启写帧线程）
        self.open()

    def close(self):
        """关闭录制器，剩余视频完全保存（/stop 强制落盘）。"""
        # 先停写帧线程（锁外）
        self._stop.set()
        if self._raw_thread is not None:
            self._raw_thread.join(timeout=3)
        if self._ai_thread is not None:
            self._ai_thread.join(timeout=3)
        with self._lock:
            self._drain_queue(self._raw_queue, self.raw_writer)
            self._drain_queue(self._ai_queue, self.ai_writer)
            if self.raw_writer is not None:
                try:
                    self.raw_writer.release()
                except Exception:
                    pass
                sz = self._file_size(self.raw_path)
                logger.info("录制结束，raw 视频已完全保存: %s 写入帧数=%d, 文件大小=%d 字节%s",
                            self.raw_path, self._raw_written, sz,
                            " [警告: 0 字节，编码器可能不可用]" if sz == 0 else "")
            if self.ai_writer is not None:
                try:
                    self.ai_writer.release()
                except Exception:
                    pass
                sz = self._file_size(self.ai_path)
                logger.info("录制结束，ai 视频已完全保存: %s 写入帧数=%d, 文件大小=%d 字节%s",
                            self.ai_path, self._ai_written, sz,
                            " [警告: 0 字节]" if sz == 0 else "")
            self.raw_writer = None
            self.ai_writer = None
            self.raw_path = None
            self.ai_path = None
            self._raw_thread = None
            self._ai_thread = None
            # 清空残留（拉流线程可能在 close 瞬间仍塞帧）
            for q in (self._raw_queue, self._ai_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def is_recording(self):
        return self.raw_writer is not None

    @property
    def path(self):
        """兼容旧调用：返回 raw 视频路径。"""
        return self.raw_path

    def set_camera(self, camera):
        """后置注入 camera 引用（用于 ai 渲染拉 metrics）。"""
        self.camera = camera

    def set_session_dir(self, session_dir: str):
        """切换 / 重设当前会话目录（/start 协调层调用）。

        必须在 open() 之前调用；调用后下一次 open() 会在新目录下创建
        raw + ai 双 writer 与文件名。
        """
        if self.is_recording:
            logger.warning("set_session_dir 在录制中调用，已忽略：%s", session_dir)
            return
        self.session_dir = session_dir
        self.videos_dir = SaveDataLayout.videos_dir(session_dir)
        self.record_dir = self.videos_dir
        os.makedirs(self.videos_dir, exist_ok=True)
        logger.info("RecordingManage 会话目录切换: %s", session_dir)
