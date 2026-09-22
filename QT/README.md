# Qt 客户端（Windows 端）

投篮动作智能评分系统的 Windows 客户端。**RK3588 盒子只做算法**（MPP 硬解 → BGR 帧 →
一路送入 AI 算法识别评分），**拷贝一份不含骨架的干净 BGR 帧**通过 HTTP 下发到本客户端，
Qt 界面在 UI 线程用 **QPainter + QImage(Format_BGR888)** 直接重建显示（**不用 OpenCV**），
并自动叠加检测框 / 17 关键点 / 火柴人骨架 / 关节角度。

## 文件结构

```
QT/
├── main.py             # 入口
├── main_window.py      # 主窗口：按钮 / 日志 / 通道状态 / 关键点表 / 角度
├── render_widget.py    # QPainter 渲染控件（帧 + 框 + 骨架 + 关键点 + 角度）
├── http_client.py      # HTTP 客户端（异常统一捕获）
├── requirements.txt
├── build_exe.bat       # pyinstaller 打包脚本
└── README.md
```

## 运行

```bash
cd QT
pip install -r requirements.txt
python main.py
```

## 打包成 exe（自行命令行）

```bash
cd QT
pip install pyinstaller
# 方式一：直接执行脚本
build_exe.bat
# 方式二：命令行
pyinstaller --noconfirm --clean --windowed --name BasketballQtClient --collect-all PyQt6 main.py

```

产物：`dist/BasketballQtClient/BasketballQtClient.exe`

## 界面功能

| 按钮 | 后端接口 | 说明 |
|------|----------|------|
| 连接 | `GET /health` | 直接连接指定 IP + 端口（默认 192.168.8.249:8899） |
| 打开摄像头 | `POST /open` | 打开 RTSP 摄像头 |
| 关闭摄像头 | `POST /close` | 关闭摄像头 |
| 开始运动 | `POST /start` | 录像 + 识别 + 预缓存 |
| 停止运动 | `POST /stop` | 停止并保存 |
| 录像 | `POST /record` | 纯录像（不识别） |
| 显示单帧图片 | `GET /frames/raw` | 显示单帧（RK3588 硬解码原始 BGR）+ 17 关键点 + 火柴人骨架 + 肩肘髋膝踝角度 |
| 分析结果 | `GET /result` | 综合得分 / 各项得分 / AI 评语 |

## 显示选项（勾选）

- **是否显示框**：球员框 + 篮球框
- **是否显示关键点**：17 关键点 + 火柴人骨架
- **修改颜色**：下拉选择常用颜色（橙色/青色/绿色/红色/黄色/白色/粉色/紫色），
  底层即对应 **HTML 十六进制色值**（如 `#FF9000`）

## 通道状态

底部实时显示：解码方式（`mpp_hard` / `cv2_soft`）、解码帧率、帧缓存长度、
会话状态、投篮计数、最近异常。

## 数据契约（与服务端约定）

- `GET /frames/raw` 返回 **裸 BGR 字节流**（`application/octet-stream`），
  宽度/高度/AI 元数据放在响应头：

```
响应头：
  X-Frame-Width: 1280
  X-Frame-Height: 720
  X-Frame-Meta: {"state":"running","shot_count":3,
                 "meta":{"player_box":[...],"ball_boxes":[...],
                         "kpts":[[x,y,conf],...],
                         "angles":{"shoulder":..,"elbow":..,"hip":..,"knee":..,"ankle":..},
                         "side":"Right"}}
响应体：
  <预览帧宽 x 预览帧高 x 3 的原始 BGR 字节流>
```

> Qt 端用 `QImage(raw, w, h, w*3, QImage.Format.Format_BGR888)` 直接重建，全程不经过
> OpenCV 的 JPEG 编码/解码。关键点 / 框 / 角度坐标均在 **预览帧** 坐标系内，
> Qt 端直接叠加绘制，无需坐标反算。骨架连线与算法端 `SKELETON_CONNECTIONS`（COCO）一致。

## 注意事项

- 首次「连接」成功后，点「打开摄像头」即开始持续拉帧预览；「显示单帧图片」为手动单帧。
- 所有网络/解析异常均在 `http_client.py` 捕获并回显到日志面板，不会导致界面崩溃。
- 帧渲染在 UI 线程（`paintEvent`），网络请求在后台 `FramePoller` / `ApiTask` 线程。
