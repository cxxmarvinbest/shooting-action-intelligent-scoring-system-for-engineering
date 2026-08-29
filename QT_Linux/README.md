# Qt 客户端（Linux 端，RK3588 本机可视化演示）

投篮动作智能评分系统的 **Linux 版 Qt 客户端**，用于在 RK3588 盒子（外接显示器 + 桌面环境）
本机做可视化演示。它与算法服务（`http_server.py`，监听 8899）**同机部署**，默认连接
`127.0.0.1:8899`。

> 与 `QT/`（Windows 端）的区别：
> 1. **框架**：Linux 端用 **PyQt5**（RK3588 的 Ubuntu 源只提供 `python3-pyqt5`，
>    PyQt6 无 aarch64 wheel、apt 源里也没有）；
> 2. 字体改为跨平台探测（`fonts.py`），不再硬编码 `Microsoft YaHei` / `Consolas`；
> 3. 默认连接地址从 `192.168.8.249` 改为本机 `127.0.0.1`，支持 `LQ_QT_HOST` / `LQ_QT_PORT`
>    环境变量覆盖。其余功能与 Windows 版一致。

## 文件结构

```
QT_Linux/
├── main.py             # 入口（支持 LQ_QT_HOST / LQ_QT_PORT 环境变量）
├── main_window.py      # 主窗口：按钮 / 日志 / 通道状态 / 关键点表 / 角度
├── render_widget.py    # QPainter 渲染控件（帧 + 框 + 骨架 + 关键点 + 角度）
├── http_client.py      # HTTP 客户端（默认 127.0.0.1:8899，异常统一捕获）
├── fonts.py            # 跨平台中文字体 / 等宽字体探测
├── requirements.txt
├── run.sh              # 启动脚本
└── README.md
```

## 部署步骤（RK3588 Linux，桌面环境）

### 1. 装中文字体（否则中文显示为方块）

```bash
sudo apt update
sudo apt install -y fonts-noto-cjk fonts-dejavu-core
```

### 2. 装依赖（优先 apt 系统包，避免 pip 拉不到 aarch64 轮子）

```bash
sudo apt install -y python3-pyqt5 python3-requests
```

> 若 pip 网络可用，也可 `pip install -r requirements.txt`（PyQt5>=5.15、requests）。
> 但板端 pip 常因 DNS/镜像问题失败（`域名解析出现暂时性错误`），故推荐 apt。

### 3. 启动（先确保算法服务已启动并监听 8899）

```bash
cd QT_Linux
./run.sh
# 或
python3 main.py
```

连接远端算法服务（算法服务在另一台机器时）：

```bash
LQ_QT_HOST=192.168.8.75 LQ_QT_PORT=8899 ./run.sh
```

## 界面功能

| 按钮 | 后端接口 | 说明 |
|------|----------|------|
| 连接 | `GET /health` | 校验 IP/端口（默认 127.0.0.1:8899） |
| 打开摄像头 | `POST /open` | 打开 RTSP 摄像头 |
| 关闭摄像头 | `POST /close` | 关闭摄像头 |
| 开始运动 | `POST /start` | 录像 + 识别 + 预缓存 |
| 停止运动 | `POST /stop` | 停止并保存 |
| 录像 | `POST /record` | 纯录像（不识别） |
| 显示单帧图片 | `GET /frames/raw` | 单帧 + 17 关键点 + 骨架 + 肩肘髋膝踝角度 |
| 分析结果 | `GET /result` | 综合得分 / 各项得分 / AI 评语 |

## 显示选项（勾选）

- **是否显示框**：球员框 + 篮球框
- **是否显示关键点**：17 关键点 + 火柴人骨架
- **修改颜色**：下拉选择常用颜色（深棕/深蓝/深绿/深红/深紫/白/黑等）

## 通道状态

底部实时显示：解码方式（`mpp_hard` / `cv2_soft`）、解码帧率、帧缓存长度、会话状态、
投篮计数、最近异常。

## 数据契约（与服务端约定）

与 `QT/`（Windows 端）完全一致，见 `QT/README.md`：

- `GET /frames/raw` 返回裸 BGR 字节流（`application/octet-stream`），宽度/高度/AI 元数据
  放在响应头 `X-Frame-Width` / `X-Frame-Height` / `X-Frame-Meta`。
- Qt 端用 `QImage(raw, w, h, w*3, QImage.Format_BGR888)` 直接重建，不经 OpenCV。

## 常见问题

- **`pip install` 报 `域名解析出现暂时性错误` / `Errno -3`**：盒子 DNS 未配好，不是 PyQt
  的问题。检查 `cat /etc/resolv.conf`，临时可 `echo 'nameserver 223.5.5.5' | sudo tee
  /etc/resolv.conf` 或改用 apt 源装系统包（推荐方案，见步骤 2）。
- **黑屏 / 无窗口**：确认有桌面环境（`echo $DISPLAY` 有输出），必要时在 `run.sh` 里取消
  `export QT_QPA_PLATFORM=xcb` 注释。
- **中文方块**：未装中文字体，执行步骤 1 的 `fonts-noto-cjk` 安装。
- **连接失败**：确认算法服务已启动并监听 8899（`ss -lntp | grep 8899`），或改
  `LQ_QT_HOST` 指向正确机器。
