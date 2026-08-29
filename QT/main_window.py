# -*- coding: utf-8 -*-
"""
主窗口（QT/main_window）
========================
职责：Qt 客户端主界面 —— 装配视频预览（FrameRenderWidget）、控制按钮、
      17 关键点/置信度/角度展示、通道状态（解码/帧率/缓存）、日志面板。

按钮与后端接口对应：
  连接      -> 校验 IP/端口（GET /health）
  打开摄像头 -> POST /open
  关闭摄像头 -> POST /close
  开始运动   -> POST /start
  停止运动   -> POST /stop
  录像       -> POST /record
  显示单帧   -> GET /frames?n=1&meta=1（含关键点/骨架/肩肘髋膝踝角度）
  分析结果   -> GET /result（综合得分/各项得分/AI 评语）

依赖：PyQt6 / http_client / render_widget
"""

import logging
import time

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QColor, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QSplitter,
    QMessageBox, QAbstractItemView, QDialog, QSizePolicy)

from http_client import ApiClient
from render_widget import (
    FrameRenderWidget, COLOR_PRESETS, KP_NAMES, KPT_CONF_THRESHOLD)

logger = logging.getLogger("qt_client")


# ----------------------------------------------------------------------
# 后台线程：轮询帧 + 通道状态
# ----------------------------------------------------------------------
class FramePoller(QThread):
    """后台轮询线程：持续 GET /frames 拉帧 + 周期性 GET /status。"""

    frame_ready = pyqtSignal(object, dict)   # (QImage, 完整响应 dict)
    status_ready = pyqtSignal(dict)          # 通道状态 dict
    conn_lost = pyqtSignal(str)              # 连接丢失提示

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self._stop = False
        self.polling = False  # 是否轮询帧（打开摄像头后置 True）
        self._lost_logged = False

    def stop(self):
        self._stop = True

    def run(self):
        tick = 0
        while not self._stop:
            if self.polling:
                # 拉 RK3588 硬解码的原始 BGR 帧，Qt 端用 QImage(Format_BGR888) 重建（不用 OpenCV）
                ok, raw, meta, err = self.client.get_frames_raw()
                if ok and raw:
                    qimg = self._bgr_to_qimage(raw, meta.get("width", 0), meta.get("height", 0))
                    if qimg and not qimg.isNull():
                        self.frame_ready.emit(qimg, meta)
                    self._lost_logged = False
                else:
                    # 区分「连接丢失」与「尚未有 AI 帧」：404 no frame 静默跳过
                    if "连接失败" in (err or "") or "连接超时" in (err or "") \
                            or "读取超时" in (err or ""):
                        if not self._lost_logged:
                            self.conn_lost.emit(err)
                            self._lost_logged = True
            # 周期性拉状态（无论是否在拉帧）
            tick += 1
            if tick >= 20:  # 20 * 50ms ≈ 1s
                tick = 0
                ok2, sdata, _ = self.client.get_status()
                if ok2 and sdata.get("code") == 0:
                    self.status_ready.emit(sdata)
            self.msleep(50)

    @staticmethod
    def _bgr_to_qimage(raw, w, h):
        """把原始 BGR 字节流重建为 QImage（Format_BGR888，深拷贝自持数据）。"""
        if not raw or w <= 0 or h <= 0:
            return None
        try:
            return QImage(raw, w, h, w * 3, QImage.Format.Format_BGR888).copy()
        except Exception:
            return None


# ----------------------------------------------------------------------
# 后台线程：一次性 API 动作（避免按钮点击阻塞 UI）
# ----------------------------------------------------------------------
class ApiTask(QThread):
    """执行一次 HTTP 动作，完成后通过信号回主线程。"""

    done = pyqtSignal(str, bool, object)  # (action_name, ok, data_or_err)

    def __init__(self, action_name, fn, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        self.fn = fn

    def run(self):
        ok, data, err = self.fn()
        self.done.emit(self.action_name, ok, data if ok else err)


# ----------------------------------------------------------------------
# 主窗口
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, host="192.168.8.249", port=8899):
        super().__init__()
        self.setWindowTitle("投篮动作智能评分 —— Qt 客户端")
        self.resize(1280, 800)

        self.client = ApiClient(host, port)
        self._tasks = []  # 持有 ApiTask 引用，防止被 GC

        # 关键点表/角度标签更新节流：画面每帧刷新，但表格(17行)每帧 setText 开销大，
        # 改为每 N 帧刷新一次，避免 UI 卡顿。
        self._frame_cnt = 0
        self._kp_update_interval = 5  # 每 5 帧（约 0.25s）刷新一次关键点表/角度

        # 后台轮询线程
        self.poller = FramePoller(self.client)
        self.poller.frame_ready.connect(self._on_frame)
        self.poller.status_ready.connect(self._on_status)
        self.poller.conn_lost.connect(self._on_conn_lost)
        self.poller.start()

        self._build_ui()
        self.log("Qt 客户端已启动，请点击「连接」接入算法盒子")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── 顶部：连接区 ──
        root.addWidget(self._build_connect_group())

        # ── 中部：视频预览 + 识别结果 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.render_widget = FrameRenderWidget()
        splitter.addWidget(self.render_widget)
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # ── 按钮区 ──
        root.addLayout(self._build_button_row())

        # ── 通道状态区 ──
        root.addLayout(self._build_channel_row())

        # ── 日志区 ──
        root.addWidget(self._build_log_group())

    def _build_connect_group(self):
        group = QGroupBox("连接设置")
        lay = QHBoxLayout(group)

        lay.addWidget(QLabel("IP:"))
        self.ip_edit = QLineEdit(self.client.host)
        self.ip_edit.setFixedWidth(140)
        lay.addWidget(self.ip_edit)

        lay.addWidget(QLabel("端口:"))
        self.port_edit = QLineEdit(str(self.client.port))
        self.port_edit.setFixedWidth(70)
        lay.addWidget(self.port_edit)

        self.btn_connect = QPushButton("连接")
        self.btn_connect.clicked.connect(self.on_connect)
        lay.addWidget(self.btn_connect)

        self.conn_status = QLabel("未连接")
        self.conn_status.setStyleSheet("color: #B22222; font-weight: bold;")  # 深红（firebrick）
        lay.addWidget(self.conn_status)
        lay.addStretch(1)
        return group

    def _build_result_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # 关键点表
        kp_group = QGroupBox("17 个关键点与置信度")
        kp_lay = QVBoxLayout(kp_group)
        self.kp_table = QTableWidget(17, 4)
        self.kp_table.setHorizontalHeaderLabels(["序号", "名称", "坐标(x,y)", "置信度"])
        for i, name in enumerate(KP_NAMES):
            self.kp_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.kp_table.setItem(i, 1, QTableWidgetItem(name))
            self.kp_table.setItem(i, 2, QTableWidgetItem("-"))
            self.kp_table.setItem(i, 3, QTableWidgetItem("-"))
        self.kp_table.verticalHeader().setVisible(False)
        self.kp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.kp_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.kp_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        kp_lay.addWidget(self.kp_table)
        lay.addWidget(kp_group, 1)

        # 角度显示
        ang_group = QGroupBox("关节角度（肩/肘/髋/膝/踝）")
        ang_lay = QGridLayout(ang_group)
        self.angle_labels = {}
        labels = [("肩", "shoulder"), ("肘", "elbow"), ("髋", "hip"),
                  ("膝", "knee"), ("踝", "ankle")]
        for idx, (name, key) in enumerate(labels):
            lb = QLabel(f"{name}: --")
            lb.setFont(QFont("Microsoft YaHei", 12))
            self.angle_labels[key] = lb
            ang_lay.addWidget(lb, idx // 3, idx % 3)
        lay.addWidget(ang_group)

        # 显示选项
        opt_group = QGroupBox("显示选项")
        opt_lay = QVBoxLayout(opt_group)
        self.chk_player_box = QCheckBox("显示人体框")
        self.chk_player_box.setChecked(True)
        self.chk_player_box.stateChanged.connect(self._on_options_changed)
        opt_lay.addWidget(self.chk_player_box)

        self.chk_ball_box = QCheckBox("显示篮球框")
        self.chk_ball_box.setChecked(True)
        self.chk_ball_box.stateChanged.connect(self._on_options_changed)
        opt_lay.addWidget(self.chk_ball_box)

        self.chk_kpts = QCheckBox("显示姿态点")
        self.chk_kpts.setChecked(True)
        self.chk_kpts.stateChanged.connect(self._on_options_changed)
        opt_lay.addWidget(self.chk_kpts)

        color_lay = QHBoxLayout()
        color_lay.addWidget(QLabel("修改颜色:"))
        self.color_combo = QComboBox()
        for name, _hex in COLOR_PRESETS:
            self.color_combo.addItem(f"{name} ({_hex})")
        self.color_combo.currentIndexChanged.connect(self._on_options_changed)
        color_lay.addWidget(self.color_combo)
        opt_lay.addLayout(color_lay)
        lay.addWidget(opt_group)

        return panel

    def _build_button_row(self):
        lay = QHBoxLayout()
        self.btn_open = QPushButton("打开摄像头")
        self.btn_close = QPushButton("关闭摄像头")
        self.btn_start = QPushButton("开始运动")
        self.btn_stop = QPushButton("停止运动")
        self.btn_record = QPushButton("开始录像")
        self.btn_record_stop = QPushButton("停止录像")
        self.btn_single = QPushButton("显示单帧图片")
        self.btn_result = QPushButton("分析结果")

        self.btn_open.clicked.connect(lambda: self._run_api("打开摄像头", self.client.open_camera))
        self.btn_close.clicked.connect(lambda: self._run_api("关闭摄像头", self.client.close_camera))
        self.btn_start.clicked.connect(lambda: self._run_api("开始运动", self.client.start_motion))
        self.btn_stop.clicked.connect(lambda: self._run_api("停止运动", self.client.stop_motion))
        self.btn_record.clicked.connect(lambda: self._run_api("开始录像", self.client.record))
        self.btn_record_stop.clicked.connect(lambda: self._run_api("停止录像", self.client.record_stop))
        self.btn_single.clicked.connect(self.on_single_frame)
        self.btn_result.clicked.connect(self.on_result)

        for b in (self.btn_open, self.btn_close, self.btn_start, self.btn_stop,
                  self.btn_record, self.btn_record_stop, self.btn_single, self.btn_result):
            lay.addWidget(b)
        lay.addStretch(1)
        return lay

    def _build_channel_row(self):
        lay = QHBoxLayout()
        self.lbl_decode = self._make_status_label("解码: --")
        self.lbl_fps = self._make_status_label("帧率: --")
        self.lbl_cache = self._make_status_label("缓存: --")
        self.lbl_state = self._make_status_label("状态: --")
        self.lbl_shot = self._make_status_label("投篮数: --")
        for lb in (self.lbl_decode, self.lbl_fps, self.lbl_cache,
                   self.lbl_state, self.lbl_shot):
            lay.addWidget(lb)
        lay.addStretch(1)
        return lay

    def _build_log_group(self):
        group = QGroupBox("日志（含异常捕获）")
        lay = QVBoxLayout(group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setFont(QFont("Consolas", 9))
        lay.addWidget(self.log_text)
        return group

    @staticmethod
    def _make_status_label(text):
        lb = QLabel(text)
        lb.setFont(QFont("Microsoft YaHei", 10))
        lb.setStyleSheet("color: #1E40AF;")  # 深蓝（通道状态）
        return lb

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def log(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        color = {"INFO": "#A6ADC8", "WARN": "#8B4513", "ERROR": "#B22222"}.get(level, "#A6ADC8")
        self.log_text.append(
            f'<span style="color:#1E40AF;">[{ts}]</span> '
            f'<span style="color:{color};">[{level}]</span> {msg}')
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    def on_connect(self):
        host = self.ip_edit.text().strip()
        port_text = self.port_edit.text().strip()
        if not host:
            self.log("连接失败：IP 不能为空", "ERROR")
            return
        try:
            port = int(port_text)
        except ValueError:
            self.log("连接失败：端口必须为整数", "ERROR")
            return
        self.client.set_endpoint(host, port)
        self.log(f"正在连接 {self.client.base_url} ...")
        self._run_api("连接", self.client.health)

    def _on_conn_lost(self, err):
        self.conn_status.setText("已断开")
        self.conn_status.setStyleSheet("color: #B22222; font-weight: bold;")  # 深红
        self.log(f"连接丢失：{err}", "ERROR")

    # ------------------------------------------------------------------
    # 按钮动作（走后台线程，不阻塞 UI）
    # ------------------------------------------------------------------
    def _run_api(self, action_name, fn):
        task = ApiTask(action_name, fn)
        task.done.connect(self._on_api_done)
        self._tasks.append(task)
        task.start()
        self.log(f"已发送请求：{action_name} ...")

    def _on_api_done(self, action_name, ok, data):
        if action_name == "连接":
            if ok:
                self.conn_status.setText(f"已连接 {self.client.host}:{self.client.port}")
                self.conn_status.setStyleSheet("color: #2E7D32; font-weight: bold;")  # 深绿
                self.log(f"连接成功：{self.client.base_url}")
            else:
                self.conn_status.setText("连接失败")
                self.conn_status.setStyleSheet("color: #B22222; font-weight: bold;")  # 深红
                self.log(f"连接失败：{data}", "ERROR")
            return

        if action_name == "显示单帧":
            if ok and isinstance(data, tuple) and len(data) == 2:
                raw, meta = data
                qimg = FramePoller._bgr_to_qimage(
                    raw, meta.get("width", 0), meta.get("height", 0))
                if qimg and not qimg.isNull():
                    self._on_frame(qimg, meta)
                    self.log("单帧显示成功（含关键点/骨架/角度）")
                    return
            self.log(f"显示单帧失败：{data}", "ERROR")
            return

        if action_name == "分析结果":
            if ok and isinstance(data, dict):
                self._show_result(data)
            else:
                self.log(f"获取分析结果失败：{data}", "ERROR")
            return

        if action_name == "打开摄像头" and ok:
            self.poller.polling = True
        if action_name == "关闭摄像头":
            self.poller.polling = False

        level = "INFO" if ok else "ERROR"
        self.log(f"{action_name}：{data if ok else '失败 - ' + str(data)}", level)

    # ------------------------------------------------------------------
    # 分析结果展示
    # ------------------------------------------------------------------
    def _show_result(self, data):
        shots = data.get("shots") or []
        shot_count = data.get("shot_count", len(shots))
        lines = [f"识别到 {shot_count} 次投篮", ""]

        if not shots:
            lines.append("暂无投篮结果（请先开始运动并完成投篮）")

        for shot in shots:
            s = shot.get("scores") or {}
            lines.append("=" * 40)
            lines.append(f"第 {shot.get('shot_idx', '-')} 投")
            lines.append(f"  综合总得分        : {s.get('final_score', '-')} / 100")
            lines.append(f"  阶段1（准备-下蹲）: {s.get('stage1_dtw', '-')} / 100")
            lines.append(f"  阶段2（蹬伸-出手）: {s.get('stage2_dtw', '-')} / 100")
            lines.append(f"  核心环节完整度    : {s.get('completeness', '-')} %")
            lines.append(f"  动力链协同       : {s.get('coordination', '-')} / 100")
            lines.append(f"  屈髋屈膝发力     : {s.get('knee_power', '-')} / 100")
            lines.append(f"  出手角度         : {s.get('release_angle', '-')} / 100")
            lines.append(f"  出手高度         : {s.get('height', '-')} / 100")
            ai = shot.get("ai_report") or shot.get("reports")
            if ai:
                lines.append("")
                lines.append("  【AI 教练评语】")
                if isinstance(ai, str):
                    for ln in ai.splitlines():
                        lines.append("  " + ln)
                else:
                    lines.append("  " + str(ai))

        text = "\n".join(lines)
        self.log("分析结果获取成功，详见弹窗")

        dlg = QDialog(self)
        dlg.setWindowTitle("投篮分析结果")
        dlg.resize(560, 640)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setFont(QFont("Microsoft YaHei", 11))
        lay.addWidget(te)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()

    # ------------------------------------------------------------------
    # 显示单帧 / 分析结果
    # ------------------------------------------------------------------
    def on_single_frame(self):
        self.log("正在获取单帧（RK3588 硬解码原始 BGR + 关键点/骨架/角度）...")

        def _fetch():
            ok, raw, meta, err = self.client.get_frames_raw()
            return ok, (raw, meta), err

        self._run_api("显示单帧", _fetch)

    def on_result(self):
        self.log("正在获取分析结果 ...")
        self._run_api("分析结果", self.client.get_result)

    # ------------------------------------------------------------------
    # 帧/状态回调（来自后台线程，经 queued connection 回到 UI 线程）
    # ------------------------------------------------------------------
    def _on_frame(self, qimage, data):
        # 画面每帧刷新（QPainter 渲染，高效）
        self.render_widget.set_frame(qimage, data)
        meta = (data.get("meta") or {}) if isinstance(data, dict) else {}
        # 关键点表(17行)/角度标签节流刷新，降低 UI 线程开销，避免卡顿
        self._frame_cnt += 1
        if self._frame_cnt % self._kp_update_interval == 0:
            self._update_kp_table(meta)
            self._update_angles(meta)

    def _on_status(self, sdata):
        ch = sdata.get("channel") or {}
        self.lbl_decode.setText(f"解码: {ch.get('decode', '--')}")
        self.lbl_fps.setText(f"帧率: {ch.get('fps', '--')} fps")
        self.lbl_cache.setText(f"缓存: {ch.get('cache_len', '--')}")
        self.lbl_state.setText(f"状态: {sdata.get('state', '--')}")
        self.lbl_shot.setText(f"投篮数: {sdata.get('shot_count', '--')}")
        if ch.get("last_error"):
            self.log(f"通道异常：{ch['last_error']}", "WARN")

    def _update_kp_table(self, meta):
        kpts = meta.get("kpts")
        if not kpts:
            for i in range(17):
                self.kp_table.item(i, 2).setText("-")
                self.kp_table.item(i, 3).setText("-")
            return
        for i in range(min(17, len(kpts))):
            p = kpts[i]
            x, y = p[0], p[1]
            conf = p[2] if len(p) > 2 else 1.0
            visible = x > 0 and y > 0 and conf >= KPT_CONF_THRESHOLD
            self.kp_table.item(i, 2).setText(
                f"({x:.0f}, {y:.0f})" if visible else "(不可见)")
            self.kp_table.item(i, 3).setText(f"{conf:.3f}")

    def _update_angles(self, meta):
        angles = meta.get("angles") or {}
        for key, lb in self.angle_labels.items():
            val = angles.get(key)
            lb.setText(f"{lb.text().split(':')[0]}: {val:.1f}°" if val is not None
                       else f"{lb.text().split(':')[0]}: --")

    def _on_options_changed(self):
        name, _hex = COLOR_PRESETS[self.color_combo.currentIndex()]
        self.render_widget.set_options(
            show_player_box=self.chk_player_box.isChecked(),
            show_ball_box=self.chk_ball_box.isChecked(),
            show_kpts=self.chk_kpts.isChecked(),
            skeleton_color=_hex)
        self.log(f"显示选项更新：人体框={self.chk_player_box.isChecked()}，"
                 f"篮球框={self.chk_ball_box.isChecked()}，"
                 f"姿态点={self.chk_kpts.isChecked()}，颜色={name}({_hex})")

    # ------------------------------------------------------------------
    # 退出
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.poller.stop()
        self.poller.wait(2000)
        super().closeEvent(event)
