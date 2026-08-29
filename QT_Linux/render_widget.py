# -*- coding: utf-8 -*-
"""
帧渲染控件（QT_Linux/render_widget）
=====================================
职责：用 QPainter（不用 cv2.imshow）把「服务端下发的干净帧 + AI 识别元数据」绘制出来。
  - 底层：RK3588 通过 HTTP 下发一张张 BGR 帧，Qt 端自动画
  - 叠加：是否显示框（球员框/篮球框）、是否显示关键点（17 点 + 火柴人骨架）、
          骨架颜色（可选几种常用色，底层即 HTML 十六进制色值）
  - 字体：经 fonts.cjk_font 选择跨平台中文字体，避免硬编码 Windows 字体

对外暴露：FrameRenderWidget(QWidget)
依赖：PyQt5 / fonts
"""

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QColor
from PyQt5.QtWidgets import QWidget

from fonts import cjk_font

# COCO 17 关键点骨架连线（与服务端 vision_algorithm/pose/pose_feature.py 一致）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]

# COCO 17 关键点名称（按序号）
KP_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
    "左髋", "右髋", "左膝", "右膝", "左踝", "右踝",
]

# 常用颜色预设（名称 -> HTML 十六进制，底层即对应 HTML 代码）
# 仅保留「深色」调色：深棕/深蓝/深绿/深红/深紫/白/黑 等，避免荧光黄/浅绿/青色等浅色字体
COLOR_PRESETS = [
    ("深棕", "#5D2F0E"),
    ("深蓝", "#1E40AF"),
    ("深绿", "#2E7D32"),
    ("深红", "#B22222"),
    ("马鞍棕", "#8B4513"),
    ("白色", "#FFFFFF"),
    ("深紫", "#4A148C"),
    ("黑色", "#1A1A1A"),
]

# 关键点置信度阈值：低于该值的关键点不绘制
KPT_CONF_THRESHOLD = 0.3


class FrameRenderWidget(QWidget):
    """帧渲染控件：绘制干净帧 + 框/骨架/关键点/角度。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 480)
        self.setAutoFillBackground(True)
        self._qimage = None       # 当前帧 QImage
        self._meta = None         # AI 识别元数据 dict
        self.show_player_box = True   # 是否显示人体框
        self.show_ball_box = True     # 是否显示篮球框
        self.show_kpts = True     # 是否显示关键点
        self.skeleton_color = "#5D2F0E"  # 骨架/关键点默认颜色（深棕）
        self.setStyleSheet("background-color: #141414;")

    # ------------------------------------------------------------------
    # 数据写入（供主窗口调用）
    # ------------------------------------------------------------------
    def set_frame(self, qimage, meta=None):
        self._qimage = qimage
        self._meta = meta
        self.update()  # 触发 paintEvent（在 UI 线程渲染）

    def set_options(self, show_player_box=None, show_ball_box=None,
                    show_kpts=None, skeleton_color=None):
        if show_player_box is not None:
            self.show_player_box = show_player_box
        if show_ball_box is not None:
            self.show_ball_box = show_ball_box
        if show_kpts is not None:
            self.show_kpts = show_kpts
        if skeleton_color is not None:
            self.skeleton_color = skeleton_color
        self.update()

    def clear(self):
        self._qimage = None
        self._meta = None
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self._qimage is None:
            self._draw_placeholder(painter)
            return

        # 1) 绘制帧（等比缩放居中）
        img_w, img_h = self._qimage.width(), self._qimage.height()
        scale, ox, oy = self._fit_transform(img_w, img_h)
        target_rect = QRectF(ox, oy, img_w * scale, img_h * scale)
        painter.drawImage(target_rect, self._qimage)

        if self._meta is None:
            return

        meta = self._meta.get("meta") if isinstance(self._meta, dict) else None
        if not meta:
            return

        # 2) 是否显示框：人体框 + 篮球框（分开控制）
        if self.show_player_box or self.show_ball_box:
            self._draw_boxes(painter, meta, scale, ox, oy)

        # 3) 是否显示关键点：骨架 + 关键点
        if self.show_kpts:
            self._draw_skeleton(painter, meta, scale, ox, oy)

        # 4) 角度文本（肩肘髋膝踝）叠加
        self._draw_angles_text(painter, meta)

    def _draw_placeholder(self, painter):
        painter.setPen(QPen(QColor("#5A5A5A")))
        painter.setFont(cjk_font(14))
        painter.drawText(self.rect(), Qt.AlignCenter,
                         "等待视频帧...\n点击「连接」并「打开摄像头」")

    def _fit_transform(self, img_w, img_h):
        """计算缩放与偏移，使图片等比缩放居中。"""
        w = self.width()
        h = self.height()
        if img_w <= 0 or img_h <= 0:
            return 1.0, 0.0, 0.0
        scale = min(w / img_w, h / img_h)
        ox = (w - img_w * scale) / 2.0
        oy = (h - img_h * scale) / 2.0
        return scale, ox, oy

    def _to_widget(self, x, y, scale, ox, oy):
        return QPointF(ox + x * scale, oy + y * scale)

    # ------------------------------------------------------------------
    # 框
    # ------------------------------------------------------------------
    def _draw_boxes(self, painter, meta, scale, ox, oy):
        if self.show_player_box:
            player_box = meta.get("player_box")
            if player_box:
                x1, y1, x2, y2 = player_box
                self._draw_rect(painter, x1, y1, x2, y2, "#5D2F0E", scale, ox, oy,
                                label="Player")  # 深棕
        if self.show_ball_box:
            for ball in meta.get("ball_boxes", []) or []:
                x1, y1, x2, y2 = ball
                self._draw_rect(painter, x1, y1, x2, y2, "#1E40AF", scale, ox, oy,
                                label="Ball")  # 深蓝

    def _draw_rect(self, painter, x1, y1, x2, y2, color, scale, ox, oy, label=None):
        p1 = self._to_widget(x1, y1, scale, ox, oy)
        p2 = self._to_widget(x2, y2, scale, ox, oy)
        rect = QRectF(p1, p2)
        pen = QPen(QColor(color), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        if label:
            painter.setPen(QPen(QColor(color)))
            painter.setFont(cjk_font(9))
            painter.drawText(QPointF(p1.x(), max(p1.y() - 4, 12)), label)

    # ------------------------------------------------------------------
    # 骨架与关键点
    # ------------------------------------------------------------------
    def _draw_skeleton(self, painter, meta, scale, ox, oy):
        kpts = meta.get("kpts")
        if not kpts:
            return
        n = len(kpts)
        color = QColor(self.skeleton_color)

        # 关键点有效性：坐标(0,0)或置信度过低视为不可见
        def visible(i):
            if i >= n:
                return False
            p = kpts[i]
            if len(p) < 3:
                return not (p[0] == 0 and p[1] == 0)
            return p[0] > 0 and p[1] > 0 and p[2] >= KPT_CONF_THRESHOLD

        # 骨架连线
        line_pen = QPen(color, 2)
        painter.setPen(line_pen)
        for a, b in SKELETON_CONNECTIONS:
            if visible(a) and visible(b):
                pa = self._to_widget(kpts[a][0], kpts[a][1], scale, ox, oy)
                pb = self._to_widget(kpts[b][0], kpts[b][1], scale, ox, oy)
                painter.drawLine(pa, pb)

        # 关键点（头部 5 点用更小半径，身体点用大半径）
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        for i in range(n):
            if not visible(i):
                continue
            p = self._to_widget(kpts[i][0], kpts[i][1], scale, ox, oy)
            r = 3.0 if i <= 4 else 5.0
            painter.drawEllipse(p, r, r)

    # ------------------------------------------------------------------
    # 角度文本
    # ------------------------------------------------------------------
    def _draw_angles_text(self, painter, meta):
        angles = meta.get("angles")
        side = meta.get("side") or "Unknown"
        painter.setFont(cjk_font(11))
        painter.setPen(QPen(QColor("#8B4513")))  # 马鞍棕（深色字体，避免荧光黄/浅色）
        y = 22
        painter.drawText(QPointF(12, y), f"侧别: {side}")
        y += 20
        if angles:
            labels = [
                ("肩", "shoulder"), ("肘", "elbow"), ("髋", "hip"),
                ("膝", "knee"), ("踝", "ankle"),
            ]
            for name, key in labels:
                if key in angles and angles[key] is not None:
                    painter.drawText(QPointF(12, y), f"{name}: {angles[key]:.1f}°")
                    y += 20
