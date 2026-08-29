# -*- coding: utf-8 -*-
"""
HTTP 服务管理（controller/http_manage）
=========================================
职责：flask HTTP API 服务（与 APP / Windows Qt 客户端通信），作为协调层装配
      StandardLibManage / CameraManage / InferenceManage，
      暴露 /open /close /record /start /pause /stop /status /frames /result 接口。

异常捕获（对应「HTTP 接口异常捕获」需求）：
  - 全局错误处理器：404（路径错误）/ 405（方法错误）/ 500（内部异常）
  - 每个路由体内 try/except，把业务异常统一映射为 {"code": -1, "msg": ...} + 状态码
  - 非法参数（如 /frames?n=xxx）、视频损坏（encode 失败）等显式校验

对外暴露：HttpManage（继承 ThreadBase，flask app.run 跑在独立子线程）
依赖：flask / cv2 / config / controller.* / common.exceptions
"""

import base64
import json
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response

from config import Config
from common.thread_base import ThreadBase
from common.exceptions import (
    HttpApiError, BaseAlgoError, classify_exception)

logger = logging.getLogger("basketball_scoring")


class HttpManage(ThreadBase):
    """HTTP API 服务（协调 camera / inference / recording 三个子线程）。"""

    def __init__(self, std_lib_mgr, camera, inference):
        super().__init__(name="HttpManage")
        self.std_lib_mgr = std_lib_mgr
        self.camera = camera
        self.inference = inference
        self.app = Flask(__name__)
        self._register_routes()
        self._register_error_handlers()

    # ------------------------------------------------------------------
    # 错误处理器（全局兜底，覆盖所有路由未捕获的异常）
    # ------------------------------------------------------------------
    def _register_error_handlers(self):
        app = self.app

        @app.errorhandler(404)
        def _handle_404(e):
            logger.warning("[HTTP] 404 路径不存在: %s %s", request.method, request.path)
            return jsonify({"code": -404, "msg": "路径不存在: " + request.path}), 404

        @app.errorhandler(405)
        def _handle_405(e):
            logger.warning("[HTTP] 405 方法不允许: %s %s", request.method, request.path)
            return jsonify({"code": -405, "msg": "方法不允许: " + request.method}), 405

        @app.errorhandler(BaseAlgoError)
        def _handle_algo_error(e):
            logger.error("[HTTP] 业务异常: %s", e)
            status = self._http_status(e)
            return jsonify({"code": e.errno, "msg": e.message,
                            "kind": e.kind}), status

        @app.errorhandler(Exception)
        def _handle_generic(e):
            # 兜底：未预期异常也返回结构化 JSON，不把 500 的 HTML 丢给客户端
            logger.error("[HTTP] 未预期异常: %s", e, exc_info=True)
            return jsonify({"code": -500, "msg": "服务内部异常: %s" % type(e).__name__}), 500

    @staticmethod
    def _http_status(exc):
        """按异常分类映射 HTTP 状态码。"""
        if isinstance(exc, HttpApiError):
            return exc.errno if 400 <= exc.errno <= 499 else 400
        return 500

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------
    def _register_routes(self):
        app = self.app

        @app.route("/health", methods=["GET"])
        def api_health():
            return jsonify({"code": 0, "msg": "ok"})

        @app.route("/open", methods=["POST"])
        def api_open():
            try:
                self.std_lib_mgr.ensure_models()
            except BaseAlgoError:
                raise
            except Exception as e:
                raise HttpApiError("模型/标准库加载失败", kind="http",
                                   cause=e) from e
            try:
                ok, msg = self.camera.open()
            except Exception as e:
                raise HttpApiError("打开摄像头失败", cause=e) from e
            if not ok:
                return jsonify({"code": -1, "msg": msg})
            # 启动推理线程：持续做帧提取 + 更新 latest_preview_frame，
            # 让 Qt 客户端「打开摄像头」即可看到预览画面，无需先「开始运动」。
            try:
                self.inference.start()
            except Exception as e:
                raise HttpApiError("启动推理线程失败", cause=e) from e
            return jsonify({"code": 0, "msg": msg})

        @app.route("/close", methods=["POST"])
        def api_close():
            try:
                # 先停推理线程，再关摄像头，避免在拉流线程退出前消费帧
                self.inference.stop()
                self.camera.recording.close()
                ok, msg = self.camera.close()
            except Exception as e:
                raise HttpApiError("关闭摄像头失败", cause=e) from e
            return jsonify({"code": 0 if ok else -1, "msg": msg})

        @app.route("/record", methods=["POST"])
        def api_record():
            # 开始录像：只开录像器写帧，不启动识别（与「开始运动」区分）
            if self.camera.state not in ("opened", "paused", "stopped"):
                raise HttpApiError("摄像头未就绪，请先 POST /open", errno=409)
            if self.camera.recording.is_recording:
                raise HttpApiError("已在录像中", errno=409)
            try:
                writer, path = self.camera.recording.open()
            except Exception as e:
                raise HttpApiError("打开录制器失败", cause=e) from e
            if writer is None:
                raise HttpApiError("录制器打开失败（视频损坏或路径不可写）", errno=500)
            self.camera.set_state("recording")
            return jsonify({"code": 0, "msg": path, "video_path": path})

        @app.route("/record/stop", methods=["POST"])
        def api_record_stop():
            # 停止录像：结束并保存（纯录像场景；运动中的录像由 /stop 负责关闭）
            if not self.camera.recording.is_recording:
                raise HttpApiError("当前未在录像", errno=409)
            path = self.camera.recording.path  # 先取路径，close 会清空 current_path
            try:
                self.camera.recording.close()
            except Exception as e:
                raise HttpApiError("停止录像失败", cause=e) from e
            # 纯录像结束后回到 opened（不影响运动识别状态）
            if self.camera.state == "recording":
                self.camera.set_state("opened")
            return jsonify({"code": 0, "msg": path, "video_path": path})

        @app.route("/start", methods=["POST"])
        def api_start():
            # 开始运动：录像 + 启用识别（推理线程已在 /open 启动，这里只打开识别开关）
            try:
                self.std_lib_mgr.ensure_models()
            except Exception as e:
                raise HttpApiError("模型/标准库加载失败", cause=e) from e
            if self.camera.state not in ("opened", "paused", "stopped"):
                raise HttpApiError("摄像头未就绪，请先 POST /open", errno=409)
            # reset_session 会创建新 segmenter 并设 recognition_active=True，启用评分
            self.inference.reset_session()
            try:
                writer, path = self.camera.recording.open()
            except Exception as e:
                raise HttpApiError("打开录制器失败", cause=e) from e
            if writer is None:
                raise HttpApiError("录制器打开失败（视频损坏或路径不可写）", errno=500)
            self.camera.set_state("running")
            return jsonify({"code": 0, "msg": path, "video_path": path})

        @app.route("/pause", methods=["POST"])
        def api_pause():
            if self.camera.state not in ("recording", "running"):
                raise HttpApiError("当前未在录制/运动中", errno=409)
            self.camera.set_state("paused")
            return jsonify({"code": 0, "msg": "paused"})

        @app.route("/stop", methods=["POST"])
        def api_stop():
            if self.camera.state not in ("recording", "running", "paused"):
                raise HttpApiError("当前未在录制/运动中", errno=409)
            path = self.camera.recording.path  # 先取路径，close 会清空 current_path
            # 关闭识别（但保留推理线程继续做帧提取，Qt 端仍能继续预览）
            self.inference.set_recognition_active(False)
            self.camera.recording.close()
            self.camera.set_state("stopped")
            return jsonify({"code": 0, "msg": path, "video_path": path})

        @app.route("/status", methods=["GET"])
        def api_status():
            return jsonify({
                "code": 0,
                "state": self.camera.state,
                "shot_count": self.inference.shot_count,
                "status_msg": self.inference.status_msg,
                "model_loaded": self.std_lib_mgr.analyzer is not None,
                "recording_path": self.camera.recording.path,
                "channel": self._channel_status(),
            })

        @app.route("/frames", methods=["GET"])
        def api_frames():
            # n：取帧数（1~10）；meta：是否附带 AI 识别元数据（关键点/置信度/角度/框）
            try:
                n = request.args.get("n", 1, type=int)
            except (TypeError, ValueError):
                raise HttpApiError("非法参数: n 必须为整数", errno=400)
            if not (1 <= n <= 10):
                raise HttpApiError("非法参数: n 取值范围 1~10", errno=400)
            with_meta = request.args.get("meta", "0") in ("1", "true", "True")

            # 附带元数据（Qt 客户端）：下发「预处理后的干净帧(544x960) + AI 元数据」，
            # 关键点/框/角度坐标与该帧坐标系一致，Qt 端可直接叠加绘制。
            if with_meta:
                preview = self.inference.latest_preview_frame
                if preview is None:
                    return jsonify({"code": -1, "msg": "no ai frame yet"}), 404
                payload = {
                    "code": 0,
                    "count": 1,
                    "frames": [self._encode_base64(preview)],
                }
                payload.update(self._latest_frame_meta(preview))
                return jsonify(payload)

            # 无元数据（旧 APP / 兼容）：返回摄像头原始帧
            frames = self.camera.recent_frames(n)
            if not frames:
                return jsonify({"code": -1, "msg": "no frame"}), 404

            # 单帧：直接返回 JPEG
            if n == 1:
                buf = self._encode_jpeg(frames[0])
                return Response(buf, mimetype="image/jpeg")

            # 多帧：JSON 返回 base64 数组
            imgs = [self._encode_base64(f) for f in frames]
            return jsonify({"code": 0, "count": len(imgs), "frames": imgs})

        @app.route("/frames/raw", methods=["GET"])
        def api_frames_raw():
            """下发 RK3588 硬件解码后的「原始 BGR 帧」（不走 OpenCV JPEG 编码）。

            返回体为裸 BGR 字节流（application/octet-stream），宽度/高度/AI 元数据
            放在响应头：X-Frame-Width / X-Frame-Height / X-Frame-Meta（ASCII JSON）。
            Qt 端用 QImage(Format_BGR888) 直接重建图像，无需 OpenCV。
            """
            preview = self.inference.latest_preview_frame
            if preview is None:
                return jsonify({"code": -1, "msg": "no ai frame yet"}), 404
            preview = np.ascontiguousarray(preview)
            h, w = preview.shape[:2]
            meta_json = json.dumps(self._latest_frame_meta(preview), ensure_ascii=True)
            resp = Response(preview.tobytes(), mimetype="application/octet-stream")
            resp.headers["X-Frame-Width"] = str(w)
            resp.headers["X-Frame-Height"] = str(h)
            resp.headers["X-Frame-Meta"] = meta_json
            resp.headers["Access-Control-Expose-Headers"] = (
                "X-Frame-Width, X-Frame-Height, X-Frame-Meta")
            return resp

        @app.route("/result", methods=["GET"])
        def api_result():
            return jsonify({
                "code": 0,
                "shot_count": len(self.inference.results),
                "shots": self.inference.results,
            })

    # ------------------------------------------------------------------
    # 帧编码与元数据组装
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_jpeg(frame):
        """把 BGR 帧编码为 JPEG 字节串；失败抛 HttpApiError（视频损坏/非法帧）。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise HttpApiError("帧为空或已损坏", errno=500)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise HttpApiError("帧 JPEG 编码失败（视频损坏）", errno=500)
        return buf.tobytes()

    @classmethod
    def _encode_base64(cls, frame):
        """把 BGR 帧编码为 JPEG 的 base64 字符串。"""
        return base64.b64encode(cls._encode_jpeg(frame)).decode("ascii")

    def _latest_frame_meta(self, frame):
        """组装最新帧的 AI 识别元数据（关键点/置信度/角度/框/状态）。"""
        meta = self.inference.latest_frame_metrics
        info = {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "state": self.camera.state,
            "shot_count": self.inference.shot_count,
            "meta": {
                "player_box": None,
                "ball_boxes": [],
                "kpts": None,
                "angles": None,
                "side": None,
            },
        }
        if meta is None:
            return info

        # 角度：肩/肘/髋/膝（踝角为独立字段）
        angles = None
        if meta.get("angles") is not None and len(meta["angles"]) >= 4:
            shoulder, elbow, hip, knee = meta["angles"][:4]
            # ankle_angle 在 pose_feature 中可能为 None（关键点未检出），
            # 用 `or 0.0` 在 None/不存在时都兜底，避免 float(None) 抛 500
            ankle_val = meta.get("ankle_angle")
            if ankle_val is None:
                ankle_val = 0.0
            angles = {
                "shoulder": round(float(shoulder), 2),
                "elbow": round(float(elbow), 2),
                "hip": round(float(hip), 2),
                "knee": round(float(knee), 2),
                "ankle": round(float(ankle_val), 2),
            }

        # 关键点（x, y, conf）三元组，无目标时为 None（置信度已内嵌于每个点）
        kpts = None
        if meta.get("kpts") is not None:
            conf = meta.get("kpt_conf")
            if conf is None:
                conf = [1.0] * len(meta["kpts"])
            kpts = [[round(float(x), 2), round(float(y), 2), round(float(c), 4)]
                    for (x, y), c in zip(meta["kpts"], conf)]

        info["meta"].update({
            "player_box": self._box_to_list(meta.get("player_box")),
            "ball_boxes": [self._box_to_list(b) for b in meta.get("ball_boxes", [])],
            "kpts": kpts,
            "angles": angles,
            "side": meta.get("side_str"),
        })
        return info

    @staticmethod
    def _box_to_list(box):
        if box is None:
            return None
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

    def _channel_status(self):
        """通道状态（解码方式 / 帧率 / 缓存 / 最近错误）。"""
        return {
            "decode": self.camera.decode_mode,
            "fps": round(self.camera.decode_fps, 2),
            "frame_idx": self.camera.latest_idx,
            "cache_len": self.camera.cache_len(),
            "last_error": self.camera.last_error,
            "recording": self.camera.recording.is_recording,
        }

    # ------------------------------------------------------------------
    # 服务线程
    # ------------------------------------------------------------------
    def _run(self):
        self.app.run(host=Config.HTTP_HOST, port=Config.HTTP_PORT, threaded=True)
