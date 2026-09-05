# -*- coding: utf-8 -*-
"""
描黑边预处理模块（tracker/preprocess）
========================================
职责：将 RTSP 摄像头横屏视频（如 1920x1080）转换为与标准视频一致的
      竖屏有效画面（544x960），消除因画幅差异导致的像素级指标口径不一致。

方案：
  1. 从横屏画面中央裁出竖幅区域，宽度 = 高度 * (544/960)
     例：1920x1080 -> 裁剪 612x1080 竖幅
  2. 将竖幅区域缩放到 544x960
  3. 若竖幅宽度不足目标比例（极端情况），左右补黑边；输入已是目标尺寸则原样返回

对外暴露：
  - LetterboxPreprocessor 类：逐帧预处理
  - letterbox_frame 函数：单帧处理
  - convert_video 函数：离线整段转换

依赖：cv2 / numpy / config
"""

import cv2
import numpy as np

from config import Config


def letterbox_frame(frame, target_w=Config.TARGET_W, target_h=Config.TARGET_H,
                    crop_x_offset=Config.CROP_X_OFFSET):
    """
    单帧预处理：横屏 -> 中央竖幅裁剪 -> 缩放到目标尺寸（必要时补黑边）。

    返回：
        (处理后的帧, 变换信息 dict)
        变换信息包含 scale / dx / dy，可用于把输出坐标反算回原始画面坐标
    """
    h, w = frame.shape[:2]

    # 已是目标尺寸：直接返回（标准视频走这里，零开销）
    if w == target_w and h == target_h:
        return frame, {'scale': 1.0, 'dx': 0, 'dy': 0}

    target_ratio = target_w / target_h  # 544/960 ≈ 0.567

    if w / h > target_ratio:
        # ── 横屏（或偏宽）画面：裁剪中间竖幅 ──
        crop_w = int(round(h * target_ratio))
        cx = w // 2 + crop_x_offset
        x1 = cx - crop_w // 2
        # 边界保护：偏移过大时贴边
        x1 = max(0, min(x1, w - crop_w))
        crop = frame[:, x1:x1 + crop_w]
        resized = cv2.resize(crop, (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)
        info = {
            'scale': h / target_h,   # 输出图 1 像素 = 原图 scale 像素
            'dx': x1,                # 原图中裁剪窗口左上角 x
            'dy': 0,
        }
        return resized, info
    else:
        # ── 竖屏但尺寸不一致：整体缩放后补黑边 ──
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        info = {'scale': 1.0 / scale, 'dx': -dx, 'dy': -dy}
        return canvas, info


class LetterboxPreprocessor:
    """逐帧在线预处理器（职责单一：只负责画面几何变换）"""

    def __init__(self, target_w=Config.TARGET_W, target_h=Config.TARGET_H,
                 crop_x_offset=Config.CROP_X_OFFSET):
        self.target_w = target_w
        self.target_h = target_h
        self.crop_x_offset = crop_x_offset

    def process(self, frame):
        """处理单帧，返回 (处理后的帧, 变换信息)"""
        return letterbox_frame(frame, self.target_w, self.target_h,
                               self.crop_x_offset)

    def need_process(self, width, height):
        """判断给定尺寸是否需要预处理（已是目标尺寸则跳过）"""
        return not (width == self.target_w and height == self.target_h)


def convert_video(src_path, dst_path, target_w=Config.TARGET_W,
                  target_h=Config.TARGET_H, crop_x_offset=Config.CROP_X_OFFSET):
    """
    离线整段视频转换（可选工具函数）：
    将摄像头横屏视频预先转成 544x960 竖屏 mp4，便于在盒子上直接复用。

    返回：dst_path（失败返回 None）
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # N5 改造：改用 FFmpegWriter（subprocess 调 ffmpeg + libx264）
    from config import Config
    from common.ffmpeg_writer import FFmpegWriter
    codec = getattr(Config, 'FFMPEG_CODEC', 'libx264')
    bitrate = getattr(Config, 'FFMPEG_BITRATE', '600k')
    out = FFmpegWriter(dst_path, target_w, target_h, fps,
                       codec=codec, bitrate=bitrate)

    pre = LetterboxPreprocessor(target_w, target_h, crop_x_offset)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(pre.process(frame)[0])

    cap.release()
    out.release()
    return dst_path
