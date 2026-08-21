# -*- coding: utf-8 -*-
"""
全局配置模块（config）
======================
集中管理 RK3588 部署版的所有可调参数：
  1. 模型文件路径（.rknn）
  2. 标准视频目录与输出目录
  3. 描黑边预处理参数（横屏摄像头视频 -> 竖屏 544x960）
  4. 检测/姿态推理阈值

说明：
  - 标准视频为 544x960 竖屏；RTSP 摄像头视频为 1920x1080 横屏。
"""

import json
import os


class Config:
    # ================= 模型路径 ===================
    # 检测模型：别 0=player, 1=ball
    DET_RKNN_PATH = os.environ.get(
        "LQ_DET_RKNN", "/home/linaro/code/intelligent_scoring_system/weights/best.rknn")
    # 姿态模型：yolov8n-pose.rknn
    POSE_RKNN_PATH = os.environ.get(
        "LQ_POSE_RKNN", "/home/linaro/code/intelligent_scoring_system/weights/yolov8n-pose_int8.rknn")

    # ================= 视频与输出目录 =================
    # 标准视频目录（544x960 竖屏）
    STANDARD_VIDEO_DIR = os.environ.get(
        "LQ_STD_DIR", "/home/linaro/code/intelligent_scoring_system/shot_clips/left_view")
    # 评分结果输出目录（分段视频 + 逐帧图）
    OUTPUT_DIR = os.environ.get(
        "LQ_OUT_DIR", "/home/linaro/code/intelligent_scoring_system/outputs")
    # 标准视频库特征缓存文件（冠军样本角度序列 + 平均出手高度，.npz）
    # 首次运行生成，之后每次评分直接读缓存，避免重复跑标准视频库推理。
    # 更换标准视频库或修改 FRAME_STRIDE 后，删除此文件重新生成。
    STANDARD_CACHE_PATH = os.environ.get(
        "LQ_STANDARD_CACHE", "/home/linaro/code/intelligent_scoring_system/cache/standard_lib.npz")

    # ================= 描黑边预处理参数 =================
    # 目标尺寸：与标准视频一致（宽 x 高）
    TARGET_W = 544
    TARGET_H = 960

    # 输入源宽高比 > 1（横屏）时启用预处理；竖屏视频（标准视频）原样通过
    # 竖幅裁剪宽度 = 输入高度 * TARGET_W / TARGET_H
    # 1920x1080 -> 裁剪 612x1080 -> 缩放到 544x960
    # 裁剪窗口横向中心偏移（像素，相对输入画面中心，正值向右）
    # 若人物不在画面正中央，可调整此值让人物进入竖幅中央
    CROP_X_OFFSET = int(os.environ.get("LQ_CROP_X_OFFSET", "0"))

    # ================= 推理阈值 =================
    DET_CONF_THRES = 0.45      # 检测置信度阈值
    DET_NMS_THRES = 0.45       # NMS IoU 阈值
    POSE_CONF_THRES = 0.3      # 姿态人体框置信度阈值

    # 篮球过滤参数
    BALL_MAX_ASPECT = 2.5      # 篮球框最大长宽比
    BALL_MAX_W_RATIO = 0.3     # 篮球框最大宽度（相对画面宽度）

    # ================= 可视化输出 =================
    OUT_VIDEO_W = 480          # 分段展示视频宽
    OUT_VIDEO_H = 640          # 分段展示视频高
    OUT_SLOW_FACTOR = 0.4      # 慢放倍率

    # ================= 隔帧采样 =================
    # 隔帧采样步长：2 表示每隔一帧分析一帧（检测+姿态+特征只做一次，速度约快一倍）。
    # 输出图片/分段视频也按此步长隔帧写出；评分里的时长/角速度用 fps/FRAME_STRIDE 折算。
    FRAME_STRIDE = int(os.environ.get("LQ_FRAME_STRIDE", "2"))

    # ================= 评分权重（综合总得分 / 阶段分数的加权叠加）=================
    # 各模块在综合总得分中的权重，会自动按权重和归一化，故无需严格等于 1。
    # 键名必须与 main.run_scoring 中组装的模块得分一致：
    #   stage1_dtw / stage2_dtw / completeness / coordination /
    #   knee_power / release_angle / height
    SCORE_WEIGHTS = {
        "stage1_dtw": 0.30,      # 阶段1（准备-下蹲）DTW 动作相似度
        "stage2_dtw": 0.30,      # 阶段2（蹬伸-出手）DTW 动作相似度
        "completeness": 0.10,    # 核心环节技术完整度
        "coordination": 0.10,    # 动力链协同与发力节奏
        "knee_power": 0.10,      # 屈髋屈膝发力与爆发性
        "release_angle": 0.05,   # 出手角度
        "height": 0.05,          # 出手高度
    }
    # 阶段1 / 阶段2 分数中，DTW 自身所占比例；剩余部分由其它模块的加权均分补足。
    # 例：0.7 -> 阶段1 = 0.7*阶段1_DTW + 0.3*其它模块加权均分。
    PHASE_DTW_RATIO = float(os.environ.get("LQ_PHASE_DTW_RATIO", "0.7"))
    # 支持通过环境变量 LQ_SCORE_WEIGHTS（JSON 字符串）整体覆盖 SCORE_WEIGHTS，便于部署调参。
    if os.environ.get("LQ_SCORE_WEIGHTS"):
        try:
            SCORE_WEIGHTS = json.loads(os.environ["LQ_SCORE_WEIGHTS"])
        except Exception:
            pass

    # ================= 实时摄像头（RTSP 高速摄像头，暂未启用）==================
    # 接入实时高速摄像头时使用（见 http_server.py 的启用步骤）。
    # H265 硬解走 RK3588 MPP，对应 pipeline._iter_rtsp_frames（当前为注释状态）。
    CAMERA_IP = "192.168.8.89"
    CAMERA_USER = "admin"
    CAMERA_PASSWORD = "siboasi123"
    CAMERA_RTSP_URL = "rtsp://admin:siboasi123@192.168.8.89:554/h264/ch1/main/av_stream"
    CAMERA_WIDTH = 1920          # 高速摄像头输出宽
    CAMERA_HEIGHT = 1080         # 高速摄像头输出高

    # ================= HTTP 实时流服务（暂未启用）==================
    HTTP_HOST = "0.0.0.0"        # 监听地址，0.0.0.0 供 APP 跨网段访问
    HTTP_PORT = 8000             # 监听端口

