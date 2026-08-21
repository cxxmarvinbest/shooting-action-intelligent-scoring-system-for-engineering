# -*- coding: utf-8 -*-
"""
HTTP 实时流服务模块（http_server）—— 实时摄像头 + 评分接口
============================================================
【暂未启用】—— 接入实时高速摄像头时，按下方「启用步骤」取消注释后使用。

摄像头信息
  ip        = 192.168.8.89
  账号      = admin
  密码      = siboasi123
  RTSP 地址 = rtsp://admin:siboasi123@192.168.8.89:554/h264/ch1/main/av_stream
  （H265 高速摄像头，建议走 RK3588 MPP 硬件解码）

接口列表
  POST /open     打开摄像头（连接 RTSP，启动拉流线程）
  POST /close    关闭摄像头（断开 RTSP，释放资源）
  POST /record   开始运动
  POST /pause    暂停运动（暂停录制，画面继续预览）
  POST /stop     停止运动（停止录制，生成录制视频文件）
  POST /analyze  开始分析（对录制视频做评分，返回评分 JSON）
  GET  /frames   查看视频帧（返回最新一帧 JPEG）
  POST /result   保存数据（后续上传云后台，当前为占位）

启用步骤
  1. 安装依赖：pip3 install flask
  2. 启用 MPP 硬解：取消 pipeline.py 顶部 mpp import 与 _iter_rtsp_frames 的注释
  3. 取消本文件「可执行实现」注释块（删掉下面这对 ''' 与结尾的 ''' 即可）
  4. 启动：python3 http_server.py
     说明：常驻进程建议进一步把「模型加载 + 标准库预加载」提到服务启动时做一次，
           analyze 只做测试视频推理 + 评分，避免每次重复 load_models。

依赖：flask / cv2 / config / main（复用 run_scoring、list_standard_videos）
"""

import logging

logger = logging.getLogger("basketball_scoring")


# =====================================================================
# 可执行实现（暂以三引号字符串整体注释，默认不生效）
# 启用方法：删除下方「三引号开始标记」这一行，以及文件末尾对应的「三引号结束标记」那一行
# =====================================================================
'''
import os
import threading
import time

import cv2

from flask import Flask, request, jsonify, Response

from config import Config
from main import run_scoring, list_standard_videos

app = Flask(__name__)


class CameraSession:
    """摄像头会话：RTSP 拉流 + 录制 + 状态管理（线程安全）"""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "closed"        # closed / opened / recording / paused / stopped
        self.latest_frame = None     # 最新一帧（GET /frames 用）
        self.recorder = None         # cv2.VideoWriter 录制器
        self.record_path = None      # 当前录制视频路径
        self.record_dir = os.path.join(Config.OUTPUT_DIR, "recordings")
        self._grab_thread = None
        self._stop_event = threading.Event()

    # ---------- 拉流线程：持续取最新帧；录制状态下同时写入 recorder ----------
    def _grab_loop(self):
        # 硬解（推荐，H265 走 RK3588 MPP）：需先在 pipeline.py 取消 _iter_rtsp_frames 注释
        # from pipeline import VideoAnalyzer
        # frames = VideoAnalyzer()._iter_rtsp_frames(Config.CAMERA_RTSP_URL)
        # for frame in frames:
        #     ...（见下方软解等价逻辑）

        # 软解兜底（H265 可能不支持，仅用于无硬解时的验证）
        cap = cv2.VideoCapture(Config.CAMERA_RTSP_URL)
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
                if self.recorder is not None and self.state == "recording":
                    self.recorder.write(frame)
        cap.release()

    def open(self):
        with self.lock:
            if self.state != "closed":
                return False, "camera already open"
            os.makedirs(self.record_dir, exist_ok=True)
            self._stop_event.clear()
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._grab_thread.start()
            self.state = "opened"
        return True, "opened"

    def close(self):
        with self.lock:
            if self.state in ("recording", "paused"):
                self._release_recorder()
            self._stop_event.set()
            self.state = "closed"
            self.latest_frame = None
        return True, "closed"

    def start_record(self):
        with self.lock:
            if self.state not in ("opened", "paused", "stopped"):
                return False, "camera not ready"
            self.record_path = os.path.join(
                self.record_dir, time.strftime("shot_%Y%m%d_%H%M%S.mp4"))
            self.recorder = cv2.VideoWriter(
                self.record_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            self.state = "recording"
        return True, self.record_path

    def pause(self):
        with self.lock:
            if self.state != "recording":
                return False, "not recording"
            self.state = "paused"
        return True, "paused"

    def stop_record(self):
        with self.lock:
            if self.state not in ("recording", "paused"):
                return False, "not recording"
            self._release_recorder()
            self.state = "stopped"
        return True, self.record_path

    def _release_recorder(self):
        if self.recorder is not None:
            self.recorder.release()
            self.recorder = None

    def latest(self):
        with self.lock:
            return self.latest_frame


# 全局会话单例（常驻进程生命周期内复用）
session = CameraSession()


@app.route("/open", methods=["POST"])
def api_open():
    ok, msg = session.open()
    return jsonify({"code": 0 if ok else -1, "msg": msg})


@app.route("/close", methods=["POST"])
def api_close():
    ok, msg = session.close()
    return jsonify({"code": 0 if ok else -1, "msg": msg})


@app.route("/record", methods=["POST"])
def api_recode():
    ok, msg = session.start_record()
    return jsonify({"code": 0 if ok else -1, "msg": msg,
                    "video_path": msg if ok else None})


@app.route("/pause", methods=["POST"])
def api_pause():
    ok, msg = session.pause()
    return jsonify({"code": 0 if ok else -1, "msg": msg})


@app.route("/stop", methods=["POST"])
def api_stop():
    ok, msg = session.stop_record()
    return jsonify({"code": 0 if ok else -1, "msg": msg,
                    "video_path": msg if ok else None})


@app.route("/analyze", methods=["POST"])
def api_analyze():
    # 分析最近一次录制；也支持请求体传 video_path 指定视频
    data = request.get_json(silent=True) or {}
    video_path = data.get("video_path") or session.record_path
    if not video_path or not os.path.exists(video_path):
        return jsonify({"code": -1, "msg": "no recorded video"})

    std_videos = list_standard_videos(Config.STANDARD_VIDEO_DIR)
    # 复用 main 的完整评分流程；常驻优化：模型/标准库提至启动时初始化，避免每次重复加载
    result = run_scoring(video_path, std_videos, Config.OUTPUT_DIR)
    return jsonify({"code": 0, "result": result})


@app.route("/frames", methods=["GET"])
def api_frames():
    frame = session.latest()
    if frame is None:
        return jsonify({"code": -1, "msg": "no frame"}), 404
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"code": -1, "msg": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/result", methods=["POST"])
def api_result():
    # 保存评分结果到云后台（占位，后续实现 HTTP 上传）
    data = request.get_json(silent=True) or {}
    # TODO: 调用云后台 API 保存 data（评分 JSON）
    logger.info("收到保存数据请求（暂为占位）: %s", list(data.keys()))
    return jsonify({"code": 0, "msg": "result saved (stub)", "received": data})


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    # 0.0.0.0 供 APP 跨网段访问；threaded 支持多请求并发
    app.run(host="0.0.0.0", port=Config.HTTP_PORT, threaded=True)
'''
