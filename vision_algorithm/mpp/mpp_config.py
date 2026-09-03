# -*- coding: utf-8 -*-
"""
MPP 硬解配置（vision_algorithm/mpp/mpp_config）
================================================
说明：这些参数原本放在 config/mpp.yaml，现按需求「合并回代码」，避免参数散落在
yaml 中（mpp 属于算法底层实现细节，随代码走更利于维护）。

字段：
  - USE_MPP_DECODE      是否使用 MPP 硬解（实时 RTSP 流必须 True，失败不静默回退软解）
  - MPP_DISPLAY_W/H     MPP 解码输出分辨率（原始大图，仅用于预览/录制）
  - MPP_QUEUE_SIZE      本地文件 MPP 解码帧队列上限（有界队列，背压防堆积）
  - MPP_SCALE_ENABLED   是否用 RGA 在解码时直接缩放到模型输入尺寸（is_mpp_scale_img）
  - MPP_SCALE_W/H       RGA 缩放目标尺寸（= 640x360 等比缩放，Python 侧再补黑边到 640x640）
  - AUTO_TRANSCODE_H265 H265 视频自动转码 H264（cv2 软解 H265 会报 ref POC 错误）
  - FFMPEG_BIN / FFPROBE_BIN  转码/探测工具

依赖：无（纯常量，可被任意模块安全 import）
"""

# ── 硬解开关 ──
USE_MPP_DECODE = True

# ── 解码输出分辨率（原始大图，用于预览/录制）──
MPP_DISPLAY_W = 1920
MPP_DISPLAY_H = 1080

# ── 本地文件解码帧队列上限（实时流不走队列，只保留最新帧）──
MPP_QUEUE_SIZE = 30

# ── RGA 解码时缩放：启用。解码阶段用 RGA 把 1920x1080 等比缩放到 640x360（方案 B，无畸变），
#    Python 侧再用 cv2.copyMakeBorder 上下补黑边拼成 640x640 送检测。
#    仅对实时 RTSP 流生效（_iter_rtsp_frames）；离线软解路径 process_video / process_video_multi
#    无 RGA 硬件，仍保留原 letterbox 逻辑，不受此开关影响。──
MPP_SCALE_ENABLED = True
MPP_SCALE_W = 640
MPP_SCALE_H = 360

# ── H265 转码 ──
AUTO_TRANSCODE_H265 = True
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
