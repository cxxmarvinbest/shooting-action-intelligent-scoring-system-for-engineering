# -*- coding: utf-8 -*-
"""
FFmpegWriter —— 基于 subprocess 调 ffmpeg 的 H.264 软件编码写入器（N5 改造）

背景 / 动机：
    RK3588 上 cv2.VideoWriter(fourcc='avc1') 会命中硬件编码器 h264_v4l2m2m；
    该设备不可用时 VideoWriter 初始化直接失败，且 OpenCV 接口无法强制选择
    libx264 软件编码器（fourcc 只决定容器标记，不决定 ffmpeg 子编码器）。

    故改用 subprocess 显式传 `-c:v libx264`，绕开硬件编码器并精确控制码率。

接口对齐 cv2.VideoWriter 的最小集：write(frame) / isOpened() / release()，
调用方只需把构造那一行换掉即可 drop-in 替换。

RK3588板端预装 ffmpeg与libx264
    sudo apt install -y ffmpeg libx264-dev

命令示例：
    ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 1280x720 -r 25 -i - \
           -an -c:v libx264 -preset veryfast \
           -b:v 600000 -maxrate 600000 -bufsize 1200000 \
           -pix_fmt yuv420p -movflags +faststart -f mp4 OUT.mp4
"""

import logging
import subprocess
import time

import cv2

logger = logging.getLogger("basketball_scoring")


def _parse_bitrate(s):
    """把 '600k' / '1.2M' 解析成 bps 整数；纯数字按 bps 原样返回。解析失败返回 None。"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    t = str(s).strip().lower()
    mult = 1
    if t.endswith("k"):
        mult = 1000
        t = t[:-1]
    elif t.endswith("m"):
        mult = 1000000
        t = t[:-1]
    try:
        return int(float(t) * mult)
    except (ValueError, TypeError):
        return None


class FFmpegWriter:
    """把 numpy BGR 帧经 stdin 管道喂给 ffmpeg，编码为 H.264 mp4。"""

    def __init__(self, path, width, height, fps,
                 codec="libx264", bitrate="600k",
                 maxrate=None, bufsize=None,
                 preset="veryfast", threads=0,
                 stderr_log=None):
        if not path or not width or not height or not fps:
            raise ValueError("path/width/height/fps 均不能为空")
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec or "libx264"
        self.preset = preset or "veryfast"

        bitrate_bps = _parse_bitrate(bitrate) or 600000
        maxrate_bps = _parse_bitrate(maxrate) or bitrate_bps
        bufsize_bps = _parse_bitrate(bufsize) or (2 * bitrate_bps)

        # stderr 重定向到文件：既消费 ffmpeg 日志（避免 pipe 缓冲满反向阻塞），
        # 又保留错误现场供启动/编码失败定位。
        self.stderr_log = stderr_log or (path + ".ffmpeg.log")
        try:
            self._stderr_file = open(self.stderr_log, "ab")
        except OSError as e:
            logger.warning("[FFmpegWriter] 无法打开 stderr 日志 %s（%s），回退 DEVNULL",
                           self.stderr_log, e)
            self._stderr_file = subprocess.DEVNULL
            self.stderr_log = None

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", self.codec,
            "-preset", self.preset,
        ]
        if int(threads) > 0:
            cmd += ["-threads", str(int(threads))]
        cmd += [
            "-b:v", str(bitrate_bps),
            "-maxrate", str(maxrate_bps),
            "-bufsize", str(bufsize_bps),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-f", "mp4",
            path,
        ]
        self._cmd = cmd
        self._written = 0
        self._broken = False
        self._size_warned = False

        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file)
        except FileNotFoundError:
            self.proc = None
            logger.error(
                "[FFmpegWriter] 找不到 ffmpeg 可执行文件，请确认已安装并在 PATH 中"
                "（RK3588: sudo apt install -y ffmpeg libx264-dev）")
            self._close_stderr()
            return

        # 启动自检：给 ffmpeg 一点启动时间，参数错/编码器缺失会立即退出
        time.sleep(0.2)
        rc = self.proc.poll()
        if rc is not None and rc != 0:
            logger.error("[FFmpegWriter] ffmpeg 启动失败（退出码=%d）: %s", rc, path)
            self._log_stderr_tail()
            self.proc = None
            self._close_stderr()

    # ------------------------------------------------------------------
    def write(self, frame):
        """写一帧（阻塞写 stdin；管道满会阻塞，由上层队列做背压丢帧）。"""
        if self.proc is None or self.proc.stdin is None or self._broken:
            return False
        if frame is None:
            return False
        try:
            h, w = frame.shape[:2]
            if (w, h) != (self.width, self.height):
                if not self._size_warned:
                    logger.warning(
                        "[FFmpegWriter] 帧尺寸不匹配（实际=%dx%d, 期望=%dx%d），"
                        "已 resize 兜底",
                        w, h, self.width, self.height)
                    self._size_warned = True
                frame = cv2.resize(frame, (self.width, self.height))
            self.proc.stdin.write(frame.tobytes())
            self.proc.stdin.flush()
            self._written += 1
            return True
        except BrokenPipeError:
            self._broken = True
            logger.error(
                "[FFmpegWriter] ffmpeg 子进程已退出（BrokenPipe），停止写入: %s",
                self.path)
            return False
        except Exception as e:  # noqa: BLE001
            self._broken = True
            logger.warning("[FFmpegWriter] 写帧失败（%s: %s）: %s",
                           type(e).__name__, e, self.path)
            return False

    def isOpened(self):
        """子进程存活即视为可用。"""
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        """发 EOF → 等 ffmpeg 完成写 trailer（含 faststart）→ 校验退出码。"""
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass
            logger.error("[FFmpegWriter] ffmpeg 超时未退出，已强制 kill: %s", self.path)
            self.proc = None
            self._close_stderr()
            return
        rc = self.proc.returncode
        if rc != 0:
            logger.error("[FFmpegWriter] ffmpeg 退出码=%d，输出可能损坏: %s（详见 %s）",
                         rc, self.path, self.stderr_log)
            self._log_stderr_tail()
        self.proc = None
        self._close_stderr()

    # ------------------------------------------------------------------
    def _close_stderr(self):
        try:
            if self._stderr_file not in (None, subprocess.DEVNULL):
                self._stderr_file.close()
        except Exception:
            pass

    def _log_stderr_tail(self, n=10):
        """把 stderr 日志尾部打印到 error 日志，用于定位启动/编码失败。"""
        if not self.stderr_log:
            return
        try:
            with open(self.stderr_log, "r", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-n:]).strip()
            if tail:
                logger.error("[FFmpegWriter] ffmpeg stderr 尾部:\n%s", tail)
        except Exception:
            pass

    @property
    def written_frames(self):
        """实际写入管道的帧数（诊断用）。"""
        return self._written
