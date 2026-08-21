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
        "LQ_OUT_DIR", "/home/linaro/code/outputs")
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

    # ================= 关键点环形缓存（多投篮切分）=================
    # 环形队列预缓存帧数：出手事件触发后回退至少要能拿到的历史帧数。
    RING_PRECACHE_FRAMES = int(os.environ.get("LQ_RING_PRECACHE", "35"))
    # 环形缓存总容量（≥ 预缓存 + 最大动作窗口）。deque(maxlen) 滚动覆盖，天然防内存耗尽。
    # 放宽到 200：一次完整投篮约 15~35 采样帧（实拍 30~70 帧@stride=2），
    # 连续多次投篮 + 间隔需留足历史，避免回退窗口被覆盖。
    RING_MAX_FRAMES = int(os.environ.get("LQ_RING_MAX", "200"))
    # 持球防抖：连续 N 帧判定为持球才确认（滤除单帧误检）。
    HOLD_DEBOUNCE_FRAMES = int(os.environ.get("LQ_HOLD_DEBOUNCE", "3"))
    # 持球超时：确认持球后 N 帧内未触发出手事件则放弃该候选，避免挂死在未闭合动作。
    HOLD_TIMEOUT_FRAMES = int(os.environ.get("LQ_HOLD_TIMEOUT", "60"))
    # 出手后回退的最大滑动窗口（反向回溯真实动作起点用），应 ≥ 预缓存。
    # 放宽到 120：保证一次 30~70 帧的完整投篮（含站直-下蹲-蹬伸-出手）全程落在窗口内，
    # 不被截断；也避免连续投篮时上一投残影干扰本投起点。
    LOOKBACK_WINDOW_FRAMES = int(os.environ.get("LQ_LOOKBACK_WINDOW", "120"))
    # 手腕最高点（出手瞬间）检测窗口：在最近 N 帧 wrist_y 中找极小值拐点。
    RELEASE_TRIGGER_WINDOW = int(os.environ.get("LQ_RELEASE_WINDOW", "5"))
    # 持球判定：球框与球员框相交判定时，球员框外扩的像素余量（吸收检测框抖动）。
    HOLD_IOU_MARGIN = int(os.environ.get("LQ_HOLD_IOU_MARGIN", "10"))

    # ---------- 动作起点检测（站直 -> 下蹲分界，替代原「球-人框不再相交」）----------
    # 终点（出手）保留现有「手腕 y 极小值拐点」逻辑；
    # 起点改为：从终点向前回溯，找「膝关节角首次低于阈值」的帧 ——
    #   且该帧之前膝关节角连续稳定高位（> 站立阈值），作为从站直到下蹲的分界点。
    # 真实投篮屈膝下蹲时膝关节角会明显 < 150°（常到 90~120°）；
    # 仅站立/举手不出手的假动作则全程 > 165°，可被「真实下蹲」门控过滤。
    KNEE_SQUAT_THRESHOLD = float(os.environ.get("LQ_KNEE_SQUAT_THR", "150"))  # 屈膝下蹲判定阈值(°)
    KNEE_STAND_MIN = float(os.environ.get("LQ_KNEE_STAND_MIN", "165"))       # 稳定站立判定阈值(°)
    KNEE_STABLE_FRAMES = int(os.environ.get("LQ_KNEE_STABLE", "5"))          # 站立需连续稳定的帧数

    # 最短有效动作段长度（采样帧数）：段长过短视为误检，直接丢弃（不评分、不输出）。
    # 实拍一次投篮约 15~35 采样帧（30~70 实拍帧@stride=2），取 8 留足余量
    # （8 采样帧 = 16 实拍帧，远小于真实投篮下限，不会误丢真实动作）。
    MIN_SHOT_FRAMES = int(os.environ.get("LQ_MIN_SHOT_FRAMES", "8"))

    # ================= H265 视频自动转码预处理 =================
    # 摄像头自带录制通常产出 H265(HEVC)，且 RTSP 录制常缺 IDR 关键帧/moov，
    # cv2 软解 H265 会报 "Could not find ref with POC" 导致打不开/花屏。
    # 开启后：ffprobe 检测到 hevc 编码时，先用 ffmpeg 转码成 H264 临时文件再分析。
    # 设 LQ_AUTO_TRANSCODE=0 可关闭（盒子无 ffmpeg 时）。
    AUTO_TRANSCODE_H265 = bool(int(os.environ.get("LQ_AUTO_TRANSCODE", "1")))
    FFMPEG_BIN = os.environ.get("LQ_FFMPEG", "ffmpeg")
    FFPROBE_BIN = os.environ.get("LQ_FFPROBE", "ffprobe")

    # ================= 评分权重（综合总得分 / 阶段分数的加权叠加）=================
    # 各模块在综合总得分中的权重，会自动按权重和归一化，故无需严格等于 1。
    # 键名必须与 main.run_scoring 中组装的模块得分一致：
    #   stage1_dtw / stage2_dtw / completeness / coordination /
    #   knee_power / release_angle / height
    SCORE_WEIGHTS = {
        "stage1_dtw": 0.30,      # 阶段1（准备-下蹲）
        "stage2_dtw": 0.30,      # 阶段2（蹬伸-出手）
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

    # ================= 离线模式（无网络 / 断网测试）=================
    # 开启后：跳过豆包（Doubao）API 调用，改用本地规则基于六项评分生成评语，
    # 保证断网情况下流程不报错、不长时间阻塞。
    # 开关方式：
    #   1) 环境变量 LQ_OFFLINE=1（部署机最方便，免改代码）
    #   2) 命令行 python3 main.py <视频> --offline
    # 注意：--no-llm 是“完全跳过评语”，--offline 是“用本地评语替代”，两者都不会联网。
    OFFLINE_MODE = bool(int(os.environ.get("LQ_OFFLINE", "0")))

    # ================= 实时摄像头（RTSP 高速摄像头，暂未启用）==================
    # 接入实时高速摄像头时使用（见 http_server.py 的启用步骤）。
    # H265 硬解走 RK3588 MPP，对应 pipeline._iter_rtsp_frames（当前为注释状态）。
    CAMERA_IP = "192.168.8.89"
    CAMERA_USER = "admin"
    CAMERA_PASSWORD = "siboasi123"
    CAMERA_RTSP_URL = "rtsp://admin:siboasi123@192.168.8.89:554/h264/ch1/main/av_stream"
    CAMERA_WIDTH = 1920          # 高速摄像头输出宽
    CAMERA_HEIGHT = 1080         # 高速摄像头输出高

    # ================= MPP 硬解（H265 视频）=================
    # 摄像头自带录制为 H265(HEVC)，用 RK3588 MPP 硬件解码，跳过软转码。
    # 注意：mpp 封装当前只硬解 HEVC，H264 文件仍走 cv2 软解。
    # MPP 硬解失败（无 mpp_player 库 / 解码异常 / 未切出投篮）时自动回退软转码+cv2。
    USE_MPP_DECODE = bool(int(os.environ.get("LQ_USE_MPP_DECODE", "1")))
    # MPP 解码输出分辨率（RGA 直接缩放，非中央裁剪；后续 preprocessor 再做中央竖幅裁剪）。
    # 应设为摄像头原始输出分辨率。
    MPP_DISPLAY_W = int(os.environ.get("LQ_MPP_DISPLAY_W", str(CAMERA_WIDTH)))
    MPP_DISPLAY_H = int(os.environ.get("LQ_MPP_DISPLAY_H", str(CAMERA_HEIGHT)))
    # MPP 解码帧队列上限（有界队列，背压防堆积丢帧）。
    MPP_QUEUE_SIZE = int(os.environ.get("LQ_MPP_QUEUE_SIZE", "30"))

    # ================= HTTP 实时流服务（暂未启用）==================
    HTTP_HOST = "192.168.8.249"       
    HTTP_PORT = 8899             # 监听端口

