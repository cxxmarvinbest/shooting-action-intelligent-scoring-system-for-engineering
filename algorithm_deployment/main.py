# -*- coding: utf-8 -*-
"""
主程序模块（main）—— RK3588 部署版
=====================================
职责：
  1. UI 交互组件（ClickableLabel / ImageViewerDialog）
  2. 后台编排线程 ProcessingThread —— 串联「检测跟踪 → 评分 → LLM 评语」三大模块
  3. 主窗口 MainWindow（界面布局、播放器、结果展示）
  4. 程序入口

依赖：detection_tracking_rknn（VideoAnalyzer）、scoring（ScoringEngine）、llm_api（LLMCoach）
"""

import os
import re
import sys
import cv2

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QScrollArea, QMessageBox, QDialog,
                             QSlider, QGraphicsView, QGraphicsScene)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from config import Config
from pipeline import VideoAnalyzer
from scoring import ScoringEngine
from llm_api import LLMCoach
import traceback

def list_standard_videos(std_dir):
    files = [f for f in os.listdir(std_dir)
             if re.fullmatch(r"shot_\d+\.mp4", f, re.IGNORECASE)]
    files.sort(key=lambda f: int(re.search(r"\d+", f).group()))
    return [os.path.join(std_dir, f) for f in files]


# ==========================================
# 1. UI组件
# ==========================================
class ClickableLabel(QLabel):
    # 传递当前图片的索引
    clicked = pyqtSignal(int)

    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class ImageViewerDialog(QDialog):
    def __init__(self, image_paths, start_index, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.current_index = start_index

        self.setWindowTitle("🔍 查看骨架大图 (按住 Ctrl + 鼠标滚轮缩放，鼠标左键拖拽)")
        self.setStyleSheet("background-color: #1E1E2E; color: white;")
        self.resize(1000, 750)

        layout = QVBoxLayout(self)

        # 使用 QGraphicsView 支持高级缩放和拖拽
        self.view = QGraphicsView()
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # 允许鼠标拖拽平移
        self.view.setStyleSheet("border: none; background-color: #11111B;")
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = self.scene.addPixmap(QPixmap())
        layout.addWidget(self.view)

        # 底部控制栏 (向左向右按键)
        control_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.clicked.connect(self.show_prev)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 16px; font-weight: bold; color: #A6E3A1;")

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.clicked.connect(self.show_next)

        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.lbl_info)
        control_layout.addWidget(self.btn_next)
        layout.addLayout(control_layout)

        # Ctrl+滚轮 缩放
        self.view.wheelEvent = self.zoom_event

        self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_index < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[self.current_index])
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.lbl_info.setText(f"第 {self.current_index + 1} 帧 / 共 {len(self.image_paths)} 帧")
            self.btn_prev.setEnabled(self.current_index > 0)
            self.btn_next.setEnabled(self.current_index < len(self.image_paths) - 1)

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_image()

    def zoom_event(self, event):
        # 必须按住 Ctrl 键才能滚动缩放，否则是上下平移
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            zoom_in_factor = 1.15  # 参数：Ctrl+滚轮放大的倍率
            zoom_out_factor = 1 / zoom_in_factor

            # 滚轮向上放大，向下缩小
            if event.angleDelta().y() > 0:
                self.view.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.view.scale(zoom_out_factor, zoom_out_factor)
        else:
            # 调用原生的滚轮事件（页面上下滚动）
            QGraphicsView.wheelEvent(self.view, event)


# ==========================================
# 2. 后台编排线程 (避免 UI 卡顿)
#    职责单一：仅负责"编排"三大模块，不含检测/评分/LLM 具体逻辑
# ==========================================
class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(float, float, float, float, str, str, str, float, float, float, float, str, float, str, float,
                          str, float, str, str)
    error = pyqtSignal(str)

    def __init__(self, test_video_path, standard_videos):
        super().__init__()
        self.test_video_path = test_video_path
        self.standard_videos = standard_videos

    def run(self):
        try:
            self.progress_updated.emit(5)

            # ── 模块一：加载 RKNN 检测/姿态模型 ──
            analyzer = VideoAnalyzer()
            analyzer.load_models()
            self.progress_updated.emit(15)

            # ── 处理标准视频库 ──
            std_seqs1, std_seqs2, std_heights = [], [], []
            for i, path in enumerate(self.standard_videos):
                s1, s2, _, _, rel_h, _ = analyzer.process_video(path.strip())
                if s1 is not None and len(s1) > 2:
                    std_seqs1.append(s1)
                if s2 is not None and len(s2) > 2:
                    std_seqs2.append(s2)
                if rel_h > 0:
                    std_heights.append(rel_h)
                self.progress_updated.emit(15 + int(35 * (i / len(self.standard_videos))))

            avg_std_height = sum(std_heights) / len(std_heights) if std_heights else 0.5

            if not std_seqs1 or not std_seqs2:
                raise ValueError("标准视频库解析失败，无法提取两段动作特征！")

            # ── 模块二：冠军样本选择 ──
            champ1 = ScoringEngine.select_champion(std_seqs1)
            champ2 = ScoringEngine.select_champion(std_seqs2)

            self.progress_updated.emit(60)

            # ── 处理测试视频（含可视化输出）──
            out_folder = Config.OUTPUT_DIR
            test_s1, test_s2, out_v1, out_v2, test_rel_h, test_metrics = analyzer.process_video(
                self.test_video_path, save_visuals=True, out_dir=out_folder)

            if test_s1 is None or test_s2 is None:
                raise ValueError("测试视频未检测到完整的下蹲和出手动作！")

            # ── 模块二：各项评分 ──
            height_score = ScoringEngine.compute_height_score(test_rel_h, avg_std_height)

            deg1, score1 = ScoringEngine.compute_dtw_distance(champ1, test_s1)
            deg2, score2 = ScoringEngine.compute_dtw_distance(champ2, test_s2)

            coord_score, coord_report = ScoringEngine.compute_coordination(test_metrics)
            video_fps = getattr(analyzer, 'current_fps', 30.0)
            knee_score, knee_report = ScoringEngine.compute_knee_power(test_metrics, fps=video_fps)
            release_score, release_report = ScoringEngine.compute_release_angle(test_metrics)
            completeness_score, completeness_report = ScoringEngine.compute_completeness(test_metrics, fps=video_fps)

            # ── 模块三：调用豆包 API 生成 AI 教练报告 ──
            self.progress_updated.emit(90)  # 提示用户正在生成 AI 报告

            coach = LLMCoach()
            ai_report = coach.generate_report(score1, score2, completeness_score,
                                              coord_score, knee_score, release_score)

            # 发送包含了 ai_report 的完整信号
            self.progress_updated.emit(100)
            self.finished.emit(score1, score2, deg1, deg2, out_folder, out_v1, out_v2,
                               height_score, test_rel_h, avg_std_height,
                               coord_score, coord_report, knee_score, knee_report,
                               release_score, release_report, completeness_score, completeness_report,
                               ai_report)

        except Exception as e:
            self.error.emit(traceback.format_exc())


# ==========================================
# 3. PyQt5 界面主程序
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能体育评测系统 - RK3588 部署版")
        self.resize(1100, 900)
        self.video_path = None
        self.image_files_paths = []

        # 标准视频库：扫描目录（shot_1.mp4 ~ shot_29.mp4）
        self.standard_videos = list_standard_videos(Config.STANDARD_VIDEO_DIR)

        self.is_playing1 = False
        self.is_playing2 = False

        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.lbl_title = QLabel("🏀 投篮动作智能评分系统 (二段式评测)")
        self.lbl_title.setObjectName("title")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)

        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("📁 选择测试视频")
        self.btn_select.clicked.connect(self.select_video)
        self.btn_start = QPushButton("🚀 开始分段智能评分")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        self.lbl_path = QLabel(f"尚未选择视频（标准视频库: {len(self.standard_videos)} 个）")
        self.lbl_path.setAlignment(Qt.AlignCenter)
        self.lbl_path.setStyleSheet("color: #aaa;")
        layout.addWidget(self.lbl_path)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

        # 视频标题
        self.lbl_video_title = QLabel("▶️ 分段姿态跟踪 (准备-下蹲 vs 蹬伸-出手)")
        self.lbl_video_title.setObjectName("subtitle")
        self.lbl_video_title.hide()
        self.scroll_layout.addWidget(self.lbl_video_title)

        video_layout = QHBoxLayout()

        # --- 视频1 容器 ---
        v1_container = QWidget()
        v1_layout = QVBoxLayout(v1_container)
        v1_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video1 = QLabel()
        self.lbl_video1.setAlignment(Qt.AlignCenter)
        self.lbl_video1.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.lbl_video1.setMinimumSize(480, 360)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setObjectName("video_slider")
        self.slider1.setCursor(Qt.PointingHandCursor)
        self.slider1.sliderMoved.connect(self.on_slider1_moved)

        self.btn_play1 = QPushButton("▶️")
        self.btn_play1.setObjectName("icon_btn")
        self.btn_play1.setCursor(Qt.PointingHandCursor)
        self.btn_play1.clicked.connect(self.toggle_video1)

        v1_ctrl_layout = QHBoxLayout()
        v1_ctrl_layout.addWidget(self.btn_play1)
        v1_ctrl_layout.addWidget(self.slider1)

        v1_layout.addWidget(self.lbl_video1)
        v1_layout.addLayout(v1_ctrl_layout)

        # --- 视频2 容器 ---
        v2_container = QWidget()
        v2_layout = QVBoxLayout(v2_container)
        v2_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video2 = QLabel()
        self.lbl_video2.setAlignment(Qt.AlignCenter)
        self.lbl_video2.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.lbl_video2.setMinimumSize(480, 360)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setObjectName("video_slider")
        self.slider2.setCursor(Qt.PointingHandCursor)
        self.slider2.sliderMoved.connect(self.on_slider2_moved)

        self.btn_play2 = QPushButton("▶️")
        self.btn_play2.setObjectName("icon_btn")
        self.btn_play2.setCursor(Qt.PointingHandCursor)
        self.btn_play2.clicked.connect(self.toggle_video2)

        v2_ctrl_layout = QHBoxLayout()
        v2_ctrl_layout.addWidget(self.btn_play2)
        v2_ctrl_layout.addWidget(self.slider2)

        v2_layout.addWidget(self.lbl_video2)
        v2_layout.addLayout(v2_ctrl_layout)

        video_layout.addWidget(v1_container)
        video_layout.addWidget(v2_container)

        self.video_container = QWidget()
        self.video_container.setLayout(video_layout)
        self.video_container.hide()
        self.scroll_layout.addWidget(self.video_container)

        # 帧图
        self.lbl_frames_title = QLabel("🎞️ 逐帧动作分析 (👉 点击图片可放大并交互)")
        self.lbl_frames_title.setObjectName("subtitle")
        self.lbl_frames_title.hide()
        self.scroll_layout.addWidget(self.lbl_frames_title)

        self.frames_scroll = QScrollArea()
        self.frames_scroll.setFixedHeight(220)
        self.frames_scroll.setWidgetResizable(True)
        self.frames_widget = QWidget()
        self.frames_layout = QHBoxLayout(self.frames_widget)
        self.frames_layout.setAlignment(Qt.AlignLeft)
        self.frames_scroll.setWidget(self.frames_widget)
        self.frames_scroll.hide()
        self.scroll_layout.addWidget(self.frames_scroll)

        # 评分标题
        self.lbl_score_title = QLabel("🏆 双阶段综合评测报告")
        self.lbl_score_title.setObjectName("subtitle")
        self.lbl_score_title.hide()
        self.scroll_layout.addWidget(self.lbl_score_title)

        self.lbl_score_result = QLabel("")
        self.lbl_score_result.setObjectName("score_box")
        self.lbl_score_result.setAlignment(Qt.AlignCenter)
        self.lbl_score_result.hide()
        self.scroll_layout.addWidget(self.lbl_score_result)

        self.lbl_completeness_score = QLabel("")
        self.lbl_completeness_score.setAlignment(Qt.AlignCenter)
        self.lbl_completeness_score.hide()
        self.scroll_layout.addWidget(self.lbl_completeness_score)

        self.lbl_height_score = QLabel("")
        self.lbl_height_score.setAlignment(Qt.AlignCenter)
        self.lbl_height_score.hide()
        self.scroll_layout.addWidget(self.lbl_height_score)

        self.lbl_coord_score = QLabel("")
        self.lbl_coord_score.setAlignment(Qt.AlignCenter)
        self.lbl_coord_score.hide()
        self.scroll_layout.addWidget(self.lbl_coord_score)

        self.lbl_knee_score = QLabel("")
        self.lbl_knee_score.setAlignment(Qt.AlignCenter)
        self.lbl_knee_score.hide()
        self.scroll_layout.addWidget(self.lbl_knee_score)

        self.lbl_release_score = QLabel("")
        self.lbl_release_score.setAlignment(Qt.AlignCenter)
        self.lbl_release_score.hide()
        self.scroll_layout.addWidget(self.lbl_release_score)

        # AI 豆包教练点评组件
        self.lbl_ai_report = QLabel("")
        self.lbl_ai_report.setAlignment(Qt.AlignLeft)
        self.lbl_ai_report.setWordWrap(True)
        self.lbl_ai_report.hide()
        self.scroll_layout.addWidget(self.lbl_ai_report)

        self.scroll_layout.addStretch()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_videos)
        self.cap1 = None
        self.cap2 = None

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E2E; }
            QWidget { background-color: #1E1E2E; color: #CDD6F4; font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
            QLabel#title { font-size: 32px; font-weight: bold; color: #A6E3A1; margin-bottom: 10px; }
            QLabel#subtitle { font-size: 18px; font-weight: bold; color: #89B4FA; margin-top: 20px; }
            QLabel#score_box { background-color: #313244; padding: 20px; border-radius: 10px; }
            QPushButton { background-color: #89B4FA; color: #1E1E2E; font-weight: bold; border-radius: 8px; padding: 10px; font-size: 15px; }
            QPushButton:hover { background-color: #74C7EC; }
            QPushButton:disabled { background-color: #45475A; color: #6C7086; }
            QPushButton#icon_btn { background-color: #313244; color: #A6E3A1; border-radius: 20px; font-size: 18px; min-width: 40px; min-height: 40px; max-width: 40px; max-height: 40px; padding: 0px; margin-top: 5px; }
            QPushButton#icon_btn:hover { background-color: #45475A; }
            QProgressBar { border: 2px solid #45475A; border-radius: 5px; text-align: center; color: white; font-weight: bold; }
            QProgressBar::chunk { background-color: #A6E3A1; }
            QScrollArea { border: none; background-color: transparent; }

            /* 视频拖拽进度条样式 */
            QSlider#video_slider::groove:horizontal {
                border-radius: 4px;
                height: 8px;
                margin: 0px;
                background-color: #313244;
            }
            QSlider#video_slider::handle:horizontal {
                background-color: #A6E3A1;
                border: none;
                height: 16px;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider#video_slider::handle:horizontal:hover {
                background-color: #89B4FA;
            }
        """)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择测试视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.lbl_path.setText(f"已选择: {path}")
            self.btn_start.setEnabled(True)

    def start_processing(self):
        self.btn_start.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.lbl_video_title.hide()
        self.video_container.hide()
        self.lbl_frames_title.hide()
        self.frames_scroll.hide()
        self.lbl_score_title.hide()
        self.lbl_score_result.hide()
        self.lbl_ai_report.hide()  # 隐藏 AI 报告组件
        self.lbl_completeness_score.hide()
        self.lbl_height_score.hide()
        self.lbl_coord_score.hide()
        self.lbl_knee_score.hide()
        self.lbl_release_score.hide()

        for i in reversed(range(self.frames_layout.count())):
            self.frames_layout.itemAt(i).widget().setParent(None)

        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        self.timer.stop()

        self.thread = ProcessingThread(self.video_path, self.standard_videos)
        self.thread.progress_updated.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()

    def show_image_dialog(self, index):
        """点击缩略图放大查看的弹窗函数"""
        dialog = ImageViewerDialog(self.image_files_paths, index, self)
        dialog.exec_()

    def on_processing_finished(self, score1, score2, deg1, deg2, frames_dir, out_vid1, out_vid2,
                               height_score, test_rel_h, avg_std_height,
                               coord_score, coord_report, knee_score, knee_report,
                               release_score, release_report, completeness_score, completeness_report,
                               ai_report):
        self.progress_bar.hide()
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)

        self.lbl_video_title.show()
        self.video_container.show()

        self.cap1 = cv2.VideoCapture(out_vid1)
        self.cap2 = cv2.VideoCapture(out_vid2)

        if self.cap1.isOpened():
            total_frames1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider1.setRange(0, max(0, total_frames1 - 1))
        if self.cap2.isOpened():
            total_frames2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider2.setRange(0, max(0, total_frames2 - 1))

        self.is_playing1 = False
        self.is_playing2 = False
        self.btn_play1.setText("▶️")
        self.btn_play2.setText("▶️")

        self.read_first_frames()

        self.lbl_frames_title.show()
        self.frames_scroll.show()

        image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        self.image_files_paths = [os.path.join(frames_dir, f) for f in image_files]

        for idx, img_path in enumerate(self.image_files_paths):
            pixmap = QPixmap(img_path).scaledToHeight(180, Qt.SmoothTransformation)
            img_label = ClickableLabel(idx)
            img_label.setPixmap(pixmap)
            img_label.setStyleSheet(
                "QLabel { border: 2px solid #45475A; border-radius: 5px; } QLabel:hover { border: 2px solid #A6E3A1; }")
            img_label.clicked.connect(self.show_image_dialog)
            self.frames_layout.addWidget(img_label)

        self.lbl_score_title.show()
        self.lbl_score_result.show()

        final_score = (score1 + score2) / 2
        color = "#A6E3A1" if final_score > 80 else ("#F9E2AF" if final_score > 60 else "#F38BA8")

        self.lbl_score_result.setStyleSheet(
            f"background-color: #313244; padding: 20px; font-size: 20px; border-radius: 10px; border: 2px solid {color};")
        self.lbl_score_result.setText(
            f"<div align='center'>🎯 <b>综合总得分：<font color='{color}' size='6'>{final_score:.1f} / 100</font></b><br><br></div>"
            f"<table width='100%' style='color:#A6ADC8;'>"
            f"<tr><td align='center'><b>阶段1：准备-下蹲</b></td><td align='center'><b>阶段2：蹬伸-出手</b></td></tr>"
            f"<tr><td align='center'>得分：{score1:.1f}</td><td align='center'>得分：{score2:.1f}</td></tr>"
            f"</table>"
        )

        # 展示“动作完成度模块”
        self.lbl_completeness_score.show()
        comp_color = "#A6E3A1" if completeness_score == 100.0 else (
            "#F9E2AF" if completeness_score > 40.0 else "#F38BA8")
        self.lbl_completeness_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {comp_color}; margin-top: 15px;"
        )
        self.lbl_completeness_score.setText(
            f"<div align='center'>📋 <b>核心环节技术完整度：<font color='{comp_color}' size='5'>{completeness_score:.1f}%</font></b><br><br></div>"
            f"{completeness_report}"
        )

        self.lbl_height_score.show()
        h_color = "#A6E3A1" if height_score > 80 else ("#F9E2AF" if height_score > 60 else "#F38BA8")
        self.lbl_height_score.setStyleSheet(
            f"background-color: #1E1E2E; padding: 15px; font-size: 16px; border-radius: 10px; border: 1px dashed {h_color}; margin-top: 15px;"
        )
        self.lbl_height_score.setText(
            f"<div align='center'>📏 <b>出手高度专项评估：<font color='{h_color}' size='5'>{height_score:.1f} / 100</font></b><br><br></div>"
            f"<table width='100%' style='color:#A6ADC8;'>"
            f"<tr><td align='center'>测试者相对出手高度：<b>{test_rel_h:.2f}</b></td>"
            f"<td align='center'>标准参考相对高度：<b>{avg_std_height:.2f}</b></td></tr>"
            f"</table>"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 相对值 = (手腕最高点位移) / 身体像素身高。<br>此项为独立指标补充参考，不计入总分。</i></div>"
        )

        self.lbl_coord_score.show()
        c_color = "#A6E3A1" if coord_score > 80 else ("#F9E2AF" if coord_score > 60 else "#F38BA8")
        self.lbl_coord_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {c_color}; margin-top: 15px;"
        )
        self.lbl_coord_score.setText(
            f"<div align='center'>🔗 <b>动力链协同与发力节奏：<font color='{c_color}' size='5'>{coord_score:.1f} / 100</font></b><br><br></div>"
            f"{coord_report}"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 评估标准：能量应从下肢平顺传导至末端。发力节点若出现明显倒置或断档将被扣分。</i></div>"
        )

        self.lbl_knee_score.show()
        k_color = "#A6E3A1" if knee_score > 80 else ("#F9E2AF" if knee_score > 60 else "#F38BA8")
        self.lbl_knee_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {k_color}; margin-top: 15px;"
        )
        self.lbl_knee_score.setText(
            f"<div align='center'>🦵 <b>屈膝发力与爆发性：<font color='{k_color}' size='5'>{knee_score:.1f} / 100</font></b><br><br></div>"
            f"{knee_report}"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 评估标准：合理的下蹲幅度与快速的蹬伸角速度是提供投篮爆发力的核心保障。</i></div>"
        )

        self.lbl_release_score.show()
        r_color = "#A6E3A1" if release_score > 80 else ("#F9E2AF" if release_score > 60 else "#F38BA8")
        self.lbl_release_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {r_color}; margin-top: 15px;"
        )
        self.lbl_release_score.setText(
            f"<div align='center'>📐 <b>出手角度评估：<font color='{r_color}' size='5'>{release_score:.1f} / 100</font></b><br><br></div>"
            f"{release_report}"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 评估标准：计算出手瞬间小臂相对于地面的夹角。理想的投篮弧线通常需要 45° - 55° 之间的出手角度。</i></div>"
        )

        # ======= 展示 AI 豆包教练点评 =======
        self.lbl_ai_report.show()
        self.lbl_ai_report.setStyleSheet(
            "background-color: #313244; padding: 18px; border-radius: 10px; border: 2px solid #89B4FA; margin-top: 15px;"
        )
        formatted_ai_text = ai_report.replace('\n', '<br>')
        self.lbl_ai_report.setText(
            f"<div align='center'>🤖 <b>AI 豆包大模型 - 智能教练指导意见：</b><br></div>"
            f"<div style='line-height:22px; margin-top:10px; color:#CDD6F4; font-size:14px;'>{formatted_ai_text}</div>"
        )

    # =============== 独立播放器控制逻辑 ===============
    def read_first_frames(self):
        if self.cap1 and self.cap1.isOpened():
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = self.cap1.read()
            if ret1:
                self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame1, self.lbl_video1))
            self.slider1.setValue(0)

        if self.cap2 and self.cap2.isOpened():
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = self.cap2.read()
            if ret2:
                self.lbl_video2.setPixmap(self.cv2_to_qpixmap(frame2, self.lbl_video2))
            self.slider2.setValue(0)

    def toggle_video1(self):
        if not self.cap1:
            return
        if self.is_playing1:
            self.is_playing1 = False
            self.btn_play1.setText("▶️")
        else:
            if self.cap1.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap1.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing1 = True
            self.btn_play1.setText("⏸️")
            if not self.timer.isActive():
                self.timer.start(60)

    def toggle_video2(self):
        if not self.cap2:
            return
        if self.is_playing2:
            self.is_playing2 = False
            self.btn_play2.setText("▶️")
        else:
            if self.cap2.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap2.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing2 = True
            self.btn_play2.setText("⏸️")
            if not self.timer.isActive():
                self.timer.start(60)

    def on_slider1_moved(self, position):
        if self.cap1:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap1.read()
            if ret:
                self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame, self.lbl_video1))

    def on_slider2_moved(self, position):
        if self.cap2:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap2.read()
            if ret:
                self.lbl_video2.setPixmap(self.cv2_to_qpixmap(frame, self.lbl_video2))

    def update_videos(self):
        if self.is_playing1 and self.cap1:
            ret1, frame1 = self.cap1.read()
            if ret1:
                self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame1, self.lbl_video1))
                self.slider1.blockSignals(True)
                self.slider1.setValue(int(self.cap1.get(cv2.CAP_PROP_POS_FRAMES)))
                self.slider1.blockSignals(False)
            else:
                self.is_playing1 = False
                self.btn_play1.setText("▶️")
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.slider1.setValue(0)

        if self.is_playing2 and self.cap2:
            ret2, frame2 = self.cap2.read()
            if ret2:
                self.lbl_video2.setPixmap(self.cv2_to_qpixmap(frame2, self.lbl_video2))
                self.slider2.blockSignals(True)
                self.slider2.setValue(int(self.cap2.get(cv2.CAP_PROP_POS_FRAMES)))
                self.slider2.blockSignals(False)
            else:
                self.is_playing2 = False
                self.btn_play2.setText("▶️")
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.slider2.setValue(0)

        if not self.is_playing1 and not self.is_playing2:
            self.timer.stop()

    def cv2_to_qpixmap(self, frame, label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def on_processing_error(self, err_msg):
        self.progress_bar.hide()
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{err_msg}")

    def closeEvent(self, event):
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
