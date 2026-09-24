## 1. 项目概述

投篮动作智能评分系统是一套基于**单目 RGB 视频输入**、运行于 **RK3588 NPU 算法盒**上的计算机视觉应用，自动完成从 **「视频采集 → 目标检测 → 人体姿态估计 → 投篮动作分段 → 多维度量化评分 → AI 教练评语」** 的完整闭环，输出面向投篮者本人的量化评分报告与改进建议，并通过 HTTP 接口与 App / 桌面客户端实时通信。

### 1.1 核心能力

| 能力 | 实现方式 |
|------|----------|
| 视频采集 | RTSP 高速摄像头（海康，192.168.8.89）＋ RK3588 MPP 硬件解码 |
| 目标检测 | YOLOv8 两类别模型（0=球员 / 1=篮球），RKNN INT8 量化推理 |
| 姿态估计 | YOLOv8-pose 17 关键点（COCO），RKNN INT8 量化推理 |
| 动作分段 | 4 态有限状态机（FSM）＋ 关键点环形缓存回溯 |
| 量化评分 | 完整度 / 动力链 / 爆发力 / 出手角度 / 出手高度 |
| AI 评语 | 火山引擎（Ark API）大模型，离线自动降级本地规则评语 |
| 通信 | Flask HTTP API（与 App / Qt 客户端通信） |

### 1.2 技术栈

| 分类 | 技术 |
|------|------|
| 语言 | Python 3.10（板端）、C++（MPP 硬解 / 检测后处理） |
| 硬件 | RK3588（三核 NPU，位掩码 0~7） |
| 推理框架 | rknn-toolkit-lite2 |
| 模型 | YOLOv8 检测（best_int8.rknn）、YOLOv8-pose（yolov8n_pose_int8_320.rknn） |
| 视觉库 | OpenCV（BGR 处理 / 可视化） |
| Web 服务 | Flask（HTTP API） |
| 视频解码 | RK3588 MPP 硬解（RGA 缩放）＋ ffmpeg/ffprobe（转码/探测） |
| 序列比对 | fastdtw ＋ scipy |
| 大模型 | 火山引擎 Ark API |
| 客户端 | PyQt6（Windows）/ PyQt5（Linux，QPainter 渲染） |
| 配置 | 多 YAML ＋ LQ_* 环境变量覆盖 |

## 2. 系统总体架构

系统采用**多线程 + 分层解耦**架构，核心链路为「拉流线程 → 推理线程 → HTTP 服务线程」三个常驻子线程协同工作。

### 2.1 实时识别数据流

```
RTSP 摄像头
   │  (RK3588 MPP 硬解 + RGA 缩放)
   ▼
CameraManage（拉流线程）
   │  维护最新帧 / 最近帧缓存
   ▼
InferenceManage（推理线程）
   │  检测 → 主球员跟踪 → 姿态 → 特征提取 → FSM 分段 → 评分 → 落盘
   ▼
HttpManage（HTTP 服务线程，Flask）
   │  /open /close /record /start /stop /status /frames /result
   ▼
App / Qt 客户端（Windows/Linux）
```

### 2.2 单帧推理内部链路

```
原图(1920×1080)
  → MPP 硬解出 BGR 预览帧(1280×720) + RGA 等比缩放图(640×360)
  → 检测模型(640×640，两类别)
  → Anti-flicker 主球员跟踪（IoU + EMA + 外推）
  → 按 player 框抠 ROI（+margin）→ letterbox 320×320 → 姿态模型(17 关键点)
  → 关键点 N3 局部补偿 → extract_pose_features（关节角度/左右侧）
  → ShotFSM 状态机 feed → 投篮段 shot_event
  → ScoringEngine 多模块评分 + LLM 教练评语
  → SaveDataWriter 异步落盘（逐帧 src/ai 图 + data.json）
```

### 2.3 分层职责

| 层 | 目录 | 职责 |
|----|------|------|
| 配置层 | `config/` | 17 个 YAML 配置 + 统一加载器 + 环境变量覆盖 |
| 公共层 | `common/` | 日志 / 异常 / 线程基类 / FFmpeg 写盘 / 存储布局 / 性能埋点 / 关键点补偿 |
| 算法层 | `vision_algorithm/` | 检测 / 姿态 / 分段 / 评分 / 标准库 / LLM / MPP 硬解 / 预处理 |
| 控制层 | `controller/` | 摄像头 / 推理 / 录制 / HTTP / 标准库预加载管理 |
| 客户端层 | `QT/`、`QT_Linux/` | Windows PyQt6 / Linux PyQt5 可视化客户端 |

## 3. 核心功能模块详解

### 3.1 视频采集与预处理

**摄像头拉流（`controller/camera_manage.py`）**

- RTSP 拉流必须走 **RK3588 MPP 硬解**。
- 断流后做**有界重连**（`RTSP_RECONNECT_ATTEMPTS=5` 次，间隔 2s），失败置 `error` 状态。
- 维护通道状态：`decode_mode`（`mpp_hard`/`cv2_soft`）、`decode_fps`、帧缓存长度、`last_error`。
- `latest_pair()` 用单锁原子返回「同一解码帧」的预览帧 + RGA 缩放图 + 帧号，避免检测坐标与预览画面错位。

**描灰边预处理（`vision_algorithm/preprocess/letterbox.py`）**

- 横屏画面 → 中央裁剪竖幅 → 缩放，消除画幅差异导致的像素级指标口径不一致。

**MPP 硬解（`vision_algorithm/mpp/`）**

- `mpp_player.cpp`（pybind11 封装），RK3588 硬件解码 H265 + RGA 硬件缩放。
- 实时流只保留最新帧（防堆积）；本地文件用有界队列背压，保证帧号连续。

### 3.2 目标检测

**双引擎设计**（由 `config/model.yaml` 的 `DET_ENGINE` 切换）：

| 引擎 | 实现 | 特点 |
|------|------|------|
| `cpp` | `cpp_det_model.py`（pybind11 C++ 后处理） | 后处理 ~0.4ms，阈值硬编码 BOX_THRESH=0.25 |

- 两类别：`0=player`（置信度阈值 0.10）、`1=basketball`（小目标放宽到 0.05）。
- 篮球后处理过滤：长宽比 `BALL_MAX_ASPECT=2.0`、宽度占比 `BALL_MAX_W_RATIO=0.4`、球心不显著低于球员框底边。

**Anti-flicker 单目标跟踪（`single_target_tracker.py`）**

轻量纯 Python 单目标跟踪层，解决主球员框时序跳变/闪断：

- 算法：IoU 匹配 + EMA 平滑 + hangover 兜底（一阶匀速外推）+ confirm 确认。
- 两条链路各一个独立实例（离线 640×640 / 实时 640×360 坐标系不同）。

### 3.3 姿态估计

- `RKNNPoseModel`：YOLOv8-pose，17 个 COCO 关键点，静态输入 320×320。
- `detect_crop`：对 player 裁剪图「保持宽高比 + 黑边」letterbox，不暴力拉伸，避免人体变形影响精度。
- 低置信关键点坐标置 0；支持「关键点置信度通道退化」时退化为坐标判定可见性。
- `extract_pose_features`：计算肩/肘/髋/膝四角度、踝角、左右侧识别、髋部中心纵坐标、像素身高等特征。

**关键点局部补偿（`common/keypoint_compensator.py`）**

- 三帧窗口 `[i-1, i, i+1]` 线性插值，补偿姿态估计的单帧孤立跳变/消失关键点。
- 严格触发条件：邻居高置信且坐标稳定、当前帧低置信/大跳变/消失、非连续多帧丢失。
- 原始 kpts 不变，仅补偿副本送 FSM；默认关闭（生产安全）。



### 3.4 AI 大模型评语

`llm_coach.py`：封装（Ark API）生成教练评语。

- 组装六项评分提示词 → 调用 Ark 对话补全接口 → 返回评语。
- **离线模式 / 网络异常**：自动降级为本地规则评语（取最高/最低项），绝不抛出、不阻塞（超时默认 30s）。

### 3.5 存储落盘（N1 重构）

`save_data_layout.py` + `save_data_writer.py`：结构化产物统一落盘到 `save_data/{日期}/{会话}/`。

```
save_data/
  YYYY-MM-DD/                      # 一级：日期目录
    YYYYMMDD_HHMMSS_userID/        # 二级：会话目录
      videos/
        01-YYYYMMDD_HHMMSS-HHMMSS_raw.mp4   # 原始视频
        01-YYYYMMDD_HHMMSS-HHMMSS_ai.mp4    # AI 渲染视频（框+骨架）
      images/
        001/                       # 第 1 次投篮
          058-src.jpg              # 原始裁剪图
          058-ai.jpg               # 骨架渲染图
          data.json                # shot_num/时间/帧/list_pose/scoring
```

- **异步写盘**：单后台线程 + 有界队列（默认 256）+ 背压丢帧，写盘 IO 不阻塞推理主链路。
- **原子写**：先写 `.tmp` 再 rename，避免进程崩溃产生半截文件。

### 3.6 实时录制（`recording_manage.py`）

- **raw/ai 双 writer**：同一帧入两个独立队列，两条写帧线程并行编码。
- **5 分钟自动 rotate**（`RECORD_ROTATE_SEC=300`），停止时强制落盘。
- **N5 改造**：改用 FFmpegWriter（subprocess 调 ffmpeg + `libx264` 软编码），绕开 RK3588 上 `cv2.VideoWriter` 命中硬件编码器 `h264_v4l2m2m` 失败的坑。

---

## 4. 评分算法体系

评分由 `ScoringEngine`（纯静态无状态方法）承担，共 **7 个评分模块**，通过 `scoring.yaml` 权重加权叠加。

### 4.1 评分权重（scoring.yaml）

| 模块 | 权重 | 说明 |
|------|------|------|
| stage1 | 0.30 | 阶段1（准备-下蹲）分数 |
| stage2 | 0.30 | 阶段2（蹬伸-出手）分数 |
| completeness | 0.10 | 核心环节技术完整度 |
| coordination | 0.10 | 动力链协同与发力节奏 |
| knee_power | 0.10 | 屈髋屈膝发力与爆发性 |
| release_angle | 0.05 | 出手角度 |
| height | 0.05 | 出手高度（相对值） |


### 4.2 各评分模块详解



**① 核心环节技术完整度（completeness）**

- 判别三个环节：下蹲蓄力（髋部重心显著下压）、蹬伸发力（蹬伸结束膝关节充分蹬直 ≥165°）、出手释放（腕过肩 + 肘伸直推球）。
- 分数 = 完成环节数 / 3 × 100。

**② 屈髋屈膝发力与爆发性（knee_power）**

- 髋/膝两关节：屈伸幅度（理想髋 65°、膝 75°）＋ 蹬伸角速度（理想髋 300°/s、膝 350°/s）。
- 每关节 = 幅度得分 × 0.5 + 速度得分 × 0.5，取两关节平均。

**③ 动力链协同与发力节奏（coordination）**

- 下蹲最低点后的蹬伸阶段，按理想顺序（髋→膝→肩→肘→腕）计算各环节启动时序与伸展峰值。
- 顺序颠倒重罚（×4），传导过慢轻罚（×2），同步窗口 2~4 帧。

**④ 出手角度（release_angle）**

- 出手瞬间（腕最高点）小臂对地夹角，理想 50°，偏差每 1° 扣 2.5 分。

**⑤ 出手高度（height）**

- 测试相对出手高度与标准平均高度的差值映射，`100 - 差值×150`。

### 4.3 单投评分流程

`report.py` 的 `score_one_shot()` 是「离线 + 实时」共用的统一评分入口，保证两者评分口径完全一致；输入数据异常统一抛 `ScoringError`。

---

## 5. 投篮动作状态机（FSM）

### 5.1 4 态串行设计

```
IDLE → HOLD → SQUAT_RAISE → OVERHEAD_RELEASE → FOLLOW → IDLE
```

| 状态 | 含义 |
|------|------|
| IDLE | 待机：未持球 / 动作链作废后复位 |
| HOLD | 持球：球贴腕、肘屈，尚未启动 |
| SQUAT_RAISE | 下蹲+上举：下肢屈膝与上肢上举并行 |
| OVERHEAD_RELEASE | 过顶+出手：手过头到球离手 |
| FOLLOW | 跟随：球出手后手臂回落（缓冲态） |

**投篮计数判定**：走完 `HOLD → SQUAT_RAISE → OVERHEAD_RELEASE` 三态即判一次投篮（`shot_count+1` 发生在进入 `OVERHEAD_RELEASE` 时）。FOLLOW 缺失不影响计数。

### 5.2 关键事件记录

- `hold_idx`：持球确认帧
- `crouch_min_idx`：下蹲最低点帧（SQUAT_RAISE 内膝角极小值）
- `overhead_idx`：手过顶帧（= OVERHEAD_RELEASE 进入帧）
- `release_idx`：球离手帧

### 5.3 出手检测（多信号 OR）

出手瞬间姿态关键点常整帧清空，故采用多路判据（任一命中即判出手）：

1. 球框抛射运动（球心 cy 快速上升，最鲁棒，不依赖腕点）；
2. 球腕距离变化率（腕点可见时）；
3. 球腕绝对距离（腕点丢时用最近有效腕点补算）；
4. 肘关节伸直（肘角快速伸直且接近伸直）。

### 5.4 容错机制

- **迟滞（hysteresis）**：连续 N 帧满足进入条件才切换（抗关键点抖动）。
- **超时兜底**：持球超时作废、跟随超时回待机、过顶超时用手过顶帧兜底切段。
- **检测丢失 watchdog**：连续无人且无球 150 帧强制回 IDLE，防止状态卡死。

---

## 6. 配置体系

### 6.1 统一配置加载器（`config/loader.py`）

- 一个 YAML 一个领域，改某类参数只动对应文件。
- 加载时把所有 YAML 顶层 key 合并为扁平 `Config` 对象，`from config import Config` + `Config.XXX` 属性式访问。
- **环境变量优先级最高**：`LQ_*` 环境变量可覆盖对应字段（部署机免改 YAML），按默认值类型自动转换。
- 路径类字段相对路径自动 join `PROJECT_ROOT`。
- 跨文件重复键显式报错阻断启动。

### 6.2 关键配置项速查

| 配置项 | 默认值 | 位置 | 说明 |
|--------|--------|------|------|
| `DET_ENGINE` | cpp | model.yaml | 检测引擎（rknn_lite/cpp） |
| `NPU_CORE_MASK` | 7 | model.yaml | NPU 三核全开 |
| `FRAME_STRIDE` | 1 | inference.yaml | 采样步长（隔帧降负载） |
| `DET_BALL_CONF_THRES` | 0.05 | det.yaml | 篮球置信度阈值（小目标放宽） |
| `ANTI_FLICKER_ENABLE` | true | det.yaml | anti-flicker 跟踪开关 |
| `POSE_CONF_THRES` | 0.3 | pose.yaml | 姿态人体框阈值 |
| `CAMERA_RTSP_URL` | rtsp://... | camera.yaml | 摄像头拉流地址 |
| `PREVIEW_WIDTH/HEIGHT` | 1280/720 | camera.yaml | 预览/录像输出分辨率 |
| `HTTP_PORT` | 8899 | http.yaml | HTTP 监听端口 |
| `RECORD_ROTATE_SEC` | 300 | recording.yaml | 视频分片周期（秒） |
| `FFMPEG_CODEC` | libx264 | recording.yaml | 软编码器 |
| `SCORE_WEIGHTS` | 见 5.1 | scoring.yaml | 评分权重 |
| `OFFLINE_MODE` | false | settings.yaml | 离线模式 |
| `COMPENSATE_ENABLED` | false | compensate.yaml | 关键点补偿开关 |
| `PERF_ENABLED` | false | perf.yaml | 性能埋点开关 |

## 7. 部署指南（RK3588）

### 7.1 环境依赖

```bash
# Python 3.10（板端）
# 安装 rknn-toolkit-lite2（瑞芯微官方 whl）
pip3 install rknn_toolkit_lite2-<版本>-cp310-cp310-linux_aarch64.whl

# 系统依赖
sudo apt install -y ffmpeg libx264-dev libopencv-dev

# Python 依赖
pip install -r requirements.txt
```

### 7.2 MPP 硬解库编译

1. 安装 pybind11（见 `vision_algorithm/mpp/mpp_player.md`）；
2. 编译 MPP：
   ```bash
   cd vision_algorithm/mpp
   mkdir build && cd build
   cmake .. && make -j8
   # 或
   python3 mpp_build_pybind.py
   ```

### 7.3 启动

```bash
# 1. 启动算法服务（监听 8899）
python pipeline.py

# 2.（可选）Linux 本机可视化
cd QT_Linux && ./run.sh

# 3. 远程 Windows 客户端连接 192.168.8.249:8899
```

### 7.4 可视化运维工具

- **宝塔**：面板管理、计划任务（磁盘清理脚本）。
- **NoMachine**：远程桌面。
- **WinSCP**：文件传输。
