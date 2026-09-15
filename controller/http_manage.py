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
import mimetypes
import os
import re
import time

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import safe_join

from config import Config
from common.thread_base import ThreadBase
from common.exceptions import (
    HttpApiError, BaseAlgoError, classify_exception)
from common.save_data_layout import SaveDataLayout
from vision_algorithm.scoring.scoring_engine import ScoringEngine

logger = logging.getLogger("basketball_scoring")


class HttpManage(ThreadBase):
    """HTTP API 服务（协调 camera / inference / recording 三个子线程）。"""

    def __init__(self, std_lib_mgr, camera, inference):
        super().__init__(name="HttpManage")
        self.std_lib_mgr = std_lib_mgr
        self.camera = camera
        self.inference = inference
        # N1：save_data 路径布局（用于 /start 时创建 session_dir）
        self.layout = SaveDataLayout(root=Config.SAVE_DATA_ROOT)
        # N1：当前会话目录（/start 创建，/stop 结束；同一时刻最多一个活跃会话）
        self.session_dir = None
        self.session_name = None
        self.session_user_id = "0000"
        self.session_start_ts = None
        # ── MQTT 客户端引用（未启用时为 None，下行指令桥 on_mqtt_command 判空）──
        self.mqtt = None
        self.app = Flask(__name__)
        # N1：把 camera 引用注入 recording（ai 视频渲染拉 AI metrics 用）
        try:
            self.camera.recording.set_camera(self.camera)
        except Exception:
            pass
        self._register_routes()
        self._register_error_handlers()
        self._register_after_request()

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

    # ------------------------------------------------------------------
    # 协议增强：统一 JSON 响应 Content-Type 补 charset
    # ------------------------------------------------------------------
    def _register_after_request(self):
        """需求第 3 点：所有 JSON 接口返回 `application/json; charset=utf-8`。

        仅在 content_type 以 application/json 开头且尚未带 charset 时补充，
        不误伤 /frames 的 image/jpeg、/frames/raw 的 application/octet-stream，
        也不影响 /media 的视频/图片字节流，保证旧接口完全兼容。
        """
        app = self.app

        @app.after_request
        def _add_json_charset(resp):
            ct = resp.headers.get("Content-Type", "")
            if ct.startswith("application/json") and "charset" not in ct:
                resp.headers["Content-Type"] = "application/json; charset=utf-8"
            return resp

    @staticmethod
    def _http_status(exc):
        """按异常分类映射 HTTP 状态码。"""
        if isinstance(exc, HttpApiError):
            return exc.errno if 400 <= exc.errno <= 499 else 400
        return 500

    # ------------------------------------------------------------------
    # 会话元数据（N1：user_id 随会话持久化到 save_data）
    # ------------------------------------------------------------------
    def _write_session_meta(self, end_ts=None):
        """把当前会话的元数据（含 user_id）写入 session 目录下的 session_meta.json。

        会话以 /start 为创建起点、/stop 为结束点；end_ts 为空时表示「进行中」，
        /stop 时补上 end_time。写入失败仅打日志，不中断主流程。
        """
        if not self.session_dir:
            return None
        import datetime as _dt
        meta = {
            "session_name": self.session_name,
            "user_id": self.session_user_id or "0000",
            "start_time": _dt.datetime.fromtimestamp(
                self.session_start_ts).strftime("%Y-%m-%d %H:%M:%S")
                if self.session_start_ts else None,
            "start_epoch": self.session_start_ts,
            "end_time": _dt.datetime.fromtimestamp(end_ts).strftime(
                "%Y-%m-%d %H:%M:%S") if end_ts else None,
            "end_epoch": end_ts,
            "save_data_root": Config.SAVE_DATA_ROOT,
            "session_dir": self.session_dir,
            "videos_dir": os.path.join(self.session_dir, "videos"),
            "images_dir": os.path.join(self.session_dir, "images"),
        }
        meta_path = os.path.join(self.session_dir, "session_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("会话元数据已写入: %s", meta_path)
            return meta_path
        except Exception as e:
            logger.error("会话元数据写入失败（%s: %s）: %s",
                         type(e).__name__, e, meta_path)
            return None

    # ------------------------------------------------------------------
    # user_id 解析（open / record / start 共用）
    # ------------------------------------------------------------------
    def _parse_user_id(self):
        """从请求体解析 user_id（兼容 JSON / form 两种入参形态）。

        解析不到（或为空字符串）返回 None；调用方按需回退到缓存值或默认 0000。
        """
        user_id = None
        try:
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                user_id = payload.get("user_id")
            if not user_id:
                user_id = request.form.get("user_id")
        except Exception:
            user_id = None
        if user_id is not None:
            user_id = str(user_id).strip()
            if not user_id:
                user_id = None
        return user_id

    def set_mqtt(self, mqtt):
        """注入 MQTT 客户端引用（未启用时为 None）。"""
        self.mqtt = mqtt

    # ------------------------------------------------------------------
    # 运动控制核心动作（HTTP 路由与 MQTT 下行指令共用）
    # ------------------------------------------------------------------
    def _start_motion_impl(self, user_id):
        """开始运动核心逻辑（无 request 依赖，user_id 由调用方传入）。

        抛 HttpApiError 表示失败；成功返回完整响应 dict。
        """
        try:
            self.std_lib_mgr.ensure_models()
        except Exception as e:
            raise HttpApiError("模型/标准库加载失败", cause=e) from e
        if self.camera.state not in ("opened", "paused", "stopped"):
            raise HttpApiError("摄像头未就绪，请先打开摄像头", errno=409)
        # 创建 save_data/{date}/{session} 目录，注入到 camera.recording 与 inference
        try:
            self.session_dir, session_name = self.layout.new_session_dir(
                user_id=user_id)
            self.session_name = session_name
            self.session_user_id = user_id or "0000"
            self.session_start_ts = time.time()
        except Exception as e:
            raise HttpApiError("创建会话目录失败", cause=e) from e
        try:
            self.camera.recording.set_session_dir(self.session_dir)
        except Exception as e:
            raise HttpApiError("注入会话目录到录制器失败", cause=e) from e
        # 先打开录制器：失败则整体中止，避免「识别已开但录像没开」的半残状态
        try:
            writer, path, ai_writer, ai_path = self.camera.recording.open()
        except Exception as e:
            raise HttpApiError("打开录制器失败", cause=e) from e
        if writer is None:
            raise HttpApiError("录制器打开失败（视频损坏或路径不可写）", errno=500)
        # 录像就绪后再启用识别（reset_session 建新 segmenter 并设 recognition_active=True）
        self.inference.reset_session(session_dir=self.session_dir)
        # 注入 user_id 给 inference（投篮事件经 MQTT 携带正确 user_id）
        self.inference.set_session_user_id(user_id)
        # 会话元数据（含 user_id）落盘，随 save_data 持久化
        self._write_session_meta()
        self.camera.set_state("running")
        return {
            "code": 200,
            "msg": path,
            "video_path": path,
            "ai_video_path": ai_path,
            "session_dir": self.session_dir,
            "session_name": session_name,
            "user_id": user_id or "0000",
        }

    def _pause_motion_impl(self):
        """暂停运动核心逻辑，返回响应 dict。"""
        if self.camera.state not in ("recording", "running"):
            raise HttpApiError("当前未在录制/运动中", errno=409)
        self.camera.set_state("paused")
        return {"code": 200, "msg": "paused"}

    def _stop_motion_impl(self):
        """停止运动核心逻辑（含会话结束完整保存），返回响应 dict。"""
        if self.camera.state not in ("recording", "running", "paused"):
            raise HttpApiError("当前未在录制/运动中", errno=409)
        path = self.camera.recording.path  # 先取路径，close 会清空 current_path
        # 1) 关闭识别（保留推理线程继续做帧提取，Qt 端仍能继续预览）
        self.inference.set_recognition_active(False)
        # 2) 强制落盘视频（release 写入 moov，未满 5 分钟也立即保存）
        try:
            self.camera.recording.close()
        except Exception as e:
            logger.error("停止运动时视频落盘失败（%s: %s）", type(e).__name__, e)
        # 3) 排空投篮逐帧图 + data.json 异步写盘队列
        try:
            self.inference.close_shot_writer()
        except Exception as e:
            logger.error("停止运动时排空写盘队列失败（%s: %s）",
                         type(e).__name__, e)
        # 4) 会话元数据补记 end_time，标记会话结束
        self._write_session_meta(end_ts=time.time())
        self.camera.set_state("stopped")
        return {
            "code": 200, "msg": path, "video_path": path,
            "session_dir": self.session_dir,
        }

    # ------------------------------------------------------------------
    # MQTT 下行指令桥（command_handler，跑在 MQTT 网络线程）
    # ------------------------------------------------------------------
    def on_mqtt_command(self, cmd, data):
        """MQTT 下行指令分发：start / stop / pause / status / query_shot_detail。

        应答统一发到 TX topic（MqttManage.publish_* 内部判 is_connected）。
        """
        mqtt_client = self.mqtt
        if mqtt_client is None:
            return
        try:
            if cmd == "status":
                self._mqtt_handle_status(mqtt_client)
            elif cmd == "query_shot_detail":
                self._mqtt_handle_query(mqtt_client, data)
            elif cmd == "start":
                self._mqtt_handle_start(mqtt_client, data)
            elif cmd == "stop":
                self._mqtt_handle_stop(mqtt_client)
            elif cmd == "pause":
                self._mqtt_handle_pause(mqtt_client)
            else:
                mqtt_client.publish_cmd_ack(cmd, -1, "未知指令: %s" % cmd)
        except Exception as e:
            logger.error("[HTTP] MQTT 指令 %s 处理异常: %s", cmd, e)
            try:
                mqtt_client.publish_cmd_ack(cmd, -1, "处理异常: %s" % e)
            except Exception:
                pass

    def _mqtt_handle_status(self, mqtt_client):
        state = getattr(self.camera, "state", "stopped") or "stopped"
        extra = {
            "camera_state": state,
            "status_msg": getattr(self.inference, "status_msg", ""),
            "shot_count": len(getattr(self.inference, "results", []) or []),
            "user_id": self.session_user_id or "0000",
        }
        mqtt_client.publish_status(state, extra)

    def _mqtt_handle_start(self, mqtt_client, data):
        user_id = str(data.get("user_id") or "").strip() \
            or self.session_user_id or "0000"
        try:
            self._start_motion_impl(user_id)
            mqtt_client.publish_cmd_ack("start", 0, "ok", {"user_id": user_id})
        except HttpApiError as e:
            mqtt_client.publish_cmd_ack("start", -1, str(e))

    def _mqtt_handle_stop(self, mqtt_client):
        try:
            self._stop_motion_impl()
            mqtt_client.publish_cmd_ack("stop", 0, "ok")
        except HttpApiError as e:
            mqtt_client.publish_cmd_ack("stop", -1, str(e))

    def _mqtt_handle_pause(self, mqtt_client):
        try:
            self._pause_motion_impl()
            mqtt_client.publish_cmd_ack("pause", 0, "ok")
        except HttpApiError as e:
            mqtt_client.publish_cmd_ack("pause", -1, str(e))

    def _mqtt_handle_query(self, mqtt_client, data):
        shot_id = data.get("shot_id")
        shots = []
        if shot_id is not None:
            sid = self._norm_shot_id(shot_id)
            shot = self._mem_shots_indexed().get(sid) if sid else None
            if shot is not None:
                shots.append(self._mqtt_shot_summary(shot, sid))
        else:
            for shot in self.inference.results:
                sid = self._mem_shot_id(shot)
                if sid is not None:
                    shots.append(self._mqtt_shot_summary(shot, sid))
        mqtt_client.publish_query_result(shots)

    def _mqtt_shot_summary(self, shot, shot_id):
        """投篮轻量摘要（id + 得分 + url，不回完整 JSON 进 MQTT）。"""
        scores = shot.get("scores") or {}
        base = Config.get("HTTP_PUBLIC_BASE_URL", "") or \
            "http://%s:%d" % (Config.get("HTTP_HOST", "127.0.0.1"),
                              int(Config.get("HTTP_PORT", 8899)))
        return {
            "shot_id": shot_id,
            "final_score": scores.get("final_score"),
            "detail_url": (base.rstrip("/") + "/result/" + shot_id)
                          if shot_id else None,
        }

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------
    def _register_routes(self):
        app = self.app

        @app.route("/health", methods=["GET"])
        def api_health():
            return jsonify({"code": 200, "msg": "ok"})

        @app.route("/open", methods=["POST"])
        def api_open():
            # 打开摄像头时立即缓存 user_id（即使后续只录像不点「开始运动」，
            # 产物也能携带正确 user_id）；传了才覆盖，否则沿用已有缓存/默认 0000
            uid = self._parse_user_id()
            if uid:
                self.session_user_id = uid
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
            return jsonify({"code": 200, "msg": msg,
                            "user_id": self.session_user_id or "0000"})

        @app.route("/close", methods=["POST"])
        def api_close():
            try:
                # 先停推理线程，再关摄像头，避免在拉流线程退出前消费帧
                self.inference.stop()
                self.camera.recording.close()
                # N1：兜底关闭异步写盘器
                try:
                    self.inference.close_shot_writer()
                except Exception:
                    pass
                ok, msg = self.camera.close()
            except Exception as e:
                raise HttpApiError("关闭摄像头失败", cause=e) from e
            return jsonify({"code": 200 if ok else -1, "msg": msg})

        @app.route("/record", methods=["POST"])
        def api_record():
            # 开始录像：只开录像器写帧，不启动识别（与「开始运动」区分）
            if self.camera.state not in ("opened", "paused", "stopped"):
                raise HttpApiError("摄像头未就绪，请先 POST /open", errno=409)
            if self.camera.recording.is_recording:
                raise HttpApiError("已在录像中", errno=409)
            # 录像时也支持 body 传 user_id（传了则覆盖缓存），缺省沿用 open 缓存的 user_id
            uid = self._parse_user_id()
            if uid:
                self.session_user_id = uid
            # 纯录像也需要一个会话目录承载 videos/ 输出（user_id 取缓存值，缺省 0000）
            if not self.session_dir:
                try:
                    self.session_dir, self.session_name = self.layout.new_session_dir(
                        user_id=self.session_user_id or "0000")
                    self.session_start_ts = time.time()
                    self.camera.recording.set_session_dir(self.session_dir)
                    self._write_session_meta()
                except Exception as e:
                    raise HttpApiError("创建录像会话目录失败", cause=e) from e
            try:
                # open() 返回 4 元组 (raw_writer, raw_path, ai_writer, ai_path)
                writer, path, ai_writer, ai_path = self.camera.recording.open()
            except Exception as e:
                raise HttpApiError("打开录制器失败", cause=e) from e
            if writer is None:
                raise HttpApiError("录制器打开失败（视频损坏或路径不可写）", errno=500)
            self.camera.set_state("recording")
            return jsonify({
                "code": 200, "msg": path, "video_path": path,
                "ai_video_path": ai_path, "session_dir": self.session_dir,
                "user_id": self.session_user_id or "0000",
            })

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
            return jsonify({"code": 200, "msg": path, "video_path": path})

        @app.route("/start", methods=["POST"])
        def api_start():
            # user_id 优先级：请求体 body > /open 缓存 > 0000
            user_id = self._parse_user_id() or self.session_user_id or "0000"
            result = self._start_motion_impl(user_id)
            return jsonify(result)

        @app.route("/pause", methods=["POST"])
        def api_pause():
            return jsonify(self._pause_motion_impl())

        @app.route("/stop", methods=["POST"])
        def api_stop():
            return jsonify(self._stop_motion_impl())

        @app.route("/status", methods=["GET"])
        def api_status():
            return jsonify({
                "code": 200,
                "state": self.camera.state,
                "shot_count": self.inference.shot_count,
                "status_msg": self.inference.status_msg,
                "model_loaded": self.std_lib_mgr.analyzer is not None,
                "recording_path": self.camera.recording.path,
                "channel": self._channel_status(),
                # N1：save_data 会话信息
                "save_data": {
                    "root": Config.SAVE_DATA_ROOT,
                    "session_dir": self.session_dir,
                    "shot_writer": self.inference.shot_writer.stats(),
                },
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
                    "code": 200,
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
            return jsonify({"code": 200, "count": len(imgs), "frames": imgs})

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

        # ── /result 大结果拆分为细粒度业务接口（HTTP 接口增强）──────────────
        # 注意：/result/summary 必须注册在 /result/<shot_id> 之前，
        # 否则 "summary" 会被当作 shot_id 吞掉。

        @app.route("/result", methods=["GET"])
        def api_result():
            shots = [self._summarize_mem_shot(s) for s in self.inference.results]
            return jsonify({
                "code": 200,
                "shot_count": len(shots),
                "shots": shots,
            })

        @app.route("/result/summary", methods=["GET"])
        def api_result_summary():
            return jsonify(self._mem_result_summary())

        @app.route("/result/<shot_id>", methods=["GET", "POST"])
        @app.route("/result/<shot_id>/<sub>", methods=["GET", "POST"])
        def api_result_shot(shot_id, sub=None):
            sid = self._norm_shot_id(shot_id)
            if sid is None:
                raise HttpApiError("非法 shot_id: %s" % shot_id, errno=400)
            shot = self._mem_shots_indexed().get(sid)
            if shot is None:
                raise HttpApiError("投篮不存在: %s" % sid, errno=404)
            detail = self._build_mem_shot_detail(shot, sid)
            return self._render_shot_sub(detail, sub)

        # ── 历史会话（磁盘回溯）──────────────────────────────────────────
        @app.route("/sessions", methods=["GET"])
        def api_sessions():
            sessions = self._scan_sessions()
            return jsonify({"code": 200, "count": len(sessions),
                            "sessions": sessions})

        @app.route("/sessions/<name>", methods=["GET"])
        def api_session_detail(name):
            sess_dir = self._find_session_dir(name)
            if sess_dir is None:
                raise HttpApiError("会话不存在: %s" % name, errno=404)
            date_name = os.path.basename(os.path.dirname(sess_dir))
            return jsonify({
                "code": 200,
                "session": self._build_session_summary(date_name, name, sess_dir),
            })

        @app.route("/sessions/<name>/shots/<shot_id>", methods=["GET", "POST"])
        @app.route("/sessions/<name>/shots/<shot_id>/<sub>",
                   methods=["GET", "POST"])
        def api_session_shot(name, shot_id, sub=None):
            sess_dir = self._find_session_dir(name)
            if sess_dir is None:
                raise HttpApiError("会话不存在: %s" % name, errno=404)
            sid = self._norm_shot_id(shot_id)
            if sid is None:
                raise HttpApiError("非法 shot_id: %s" % shot_id, errno=400)
            shot_dir = os.path.join(SaveDataLayout.images_dir(sess_dir), sid)
            if not os.path.isdir(shot_dir):
                raise HttpApiError("投篮不存在: %s" % sid, errno=404)
            detail = self._build_disk_shot_detail(shot_dir, sid, sess_dir)
            return self._render_shot_sub(detail, sub)

        # ── 受控静态资源出口（图片/视频下载）──────────────────────────────
        @app.route("/media/<path:relpath>", methods=["GET"])
        def api_media(relpath):
            # safe_join 限制在 SAVE_DATA_ROOT 下，防路径穿越
            full = safe_join(os.path.abspath(Config.SAVE_DATA_ROOT), relpath)
            if full is None or not os.path.isfile(full):
                raise HttpApiError("资源不存在: %s" % relpath, errno=404)
            mimetype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            # ?download=1 触发下载到用户终端；否则浏览器直接打开
            as_attach = request.args.get("download") in ("1", "true", "True")
            return send_file(full, mimetype=mimetype,
                             as_attachment=as_attach, conditional=True)

    # ------------------------------------------------------------------
    # result / sessions / media 公共 helper（HTTP 接口增强）
    # ------------------------------------------------------------------
    def _public_base(self):
        """对外回显的基础 URL：优先 HTTP_PUBLIC_BASE_URL，否则 request.host_url。"""
        base = Config.get("HTTP_PUBLIC_BASE_URL", "")
        if base:
            return base.rstrip("/") + "/"
        return request.host_url.rstrip("/") + "/"

    def _rel_to_save_root(self, abs_path):
        """把 save_data 下绝对路径转相对 SAVE_DATA_ROOT 路径（越界返回 None）。"""
        if not abs_path:
            return None
        try:
            rel = os.path.relpath(abs_path, os.path.abspath(Config.SAVE_DATA_ROOT))
        except ValueError:
            return None
        if rel == ".." or rel.startswith(".." + os.sep):
            return None
        return rel

    def _media_url(self, rel_path):
        """把 save_data 相对路径转成可访问的 /media/<path> HTTP URL。"""
        if not rel_path:
            return None
        rel = rel_path.replace("\\", "/").lstrip("/")
        return self._public_base() + "media/" + rel

    @staticmethod
    def _norm_shot_id(shot_id):
        """把 shot_id 归一化为三位目录编号字符串（如 "001"），非法返回 None。

        决策 2：详情接口统一用目录编号当 shot_id；兼容 "1" / "001" / 整数 1。
        """
        try:
            n = int(str(shot_id).strip())
        except (TypeError, ValueError):
            return None
        if n < 1:
            return None
        return "%03d" % n

    @staticmethod
    def _mem_shot_id(shot):
        """内存 shot -> 目录编号（三位）；优先 save_data.shot_idx，兜底 FSM shot_idx。"""
        sd = shot.get("save_data") or {}
        v = sd.get("shot_idx")
        if v is not None:
            try:
                return "%03d" % int(v)
            except (TypeError, ValueError):
                pass
        si = shot.get("shot_idx")
        if si is not None:
            try:
                return "%03d" % int(si)
            except (TypeError, ValueError):
                return None
        return None

    def _mem_shots_indexed(self):
        """把内存 results 按目录编号建立索引 {dir_idx(三位): shot}。"""
        idx = {}
        for shot in self.inference.results:
            sid = self._mem_shot_id(shot)
            if sid is not None:
                idx[sid] = shot
        return idx

    def _scan_shot_frames(self, shot_dir):
        """扫描投篮目录下所有 jpg，返回 [{file, url}]（按文件名升序）。"""
        frames = []
        if not shot_dir or not os.path.isdir(shot_dir):
            return frames
        for name in sorted(os.listdir(shot_dir)):
            if name.lower().endswith(".jpg"):
                rel = self._rel_to_save_root(os.path.join(shot_dir, name))
                frames.append({
                    "file": name,
                    "url": self._media_url(rel) if rel else None,
                })
        return frames

    def _scan_session_videos(self, session_dir):
        """扫描会话 videos/ 下所有 mp4，返回 [{file, url, size}]。"""
        videos = []
        if not session_dir:
            return videos
        videos_dir = os.path.join(session_dir, "videos")
        if not os.path.isdir(videos_dir):
            return videos
        for name in sorted(os.listdir(videos_dir)):
            if name.lower().endswith(".mp4"):
                full = os.path.join(videos_dir, name)
                rel = self._rel_to_save_root(full)
                videos.append({
                    "file": name,
                    "url": self._media_url(rel) if rel else None,
                    "size": os.path.getsize(full) if os.path.isfile(full) else 0,
                })
        return videos

    def _load_shot_data_json(self, shot_dir):
        """读取投篮 data.json；不存在/损坏返回 None。"""
        if not shot_dir:
            return None
        path = SaveDataLayout.shot_data_json_path(shot_dir)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_session_meta(self, session_dir):
        """读取会话 session_meta.json；不存在/损坏返回 {}。"""
        path = os.path.join(session_dir, "session_meta.json")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _scores_from_disk_scoring(scoring):
        """从 data.json 的 scoring 字段重算 final_score（决策 4：combine_scores 重算）。

        注意：data.json 的 stage1/stage2 存的是「阶段加权分」
        （ratio*dtw + (1-ratio)*aux_weighted），与实时 final_score 使用的
        「原始 dtw 分」口径略有差异，故重算值与实时值可能有细微偏差（通常 < 1 分）。
        这是「历史数据未落原始 dtw 分」情况下的既定近似。
        """
        scoring = scoring or {}

        def _f(k, d=0.0):
            try:
                return float(scoring.get(k, d))
            except (TypeError, ValueError):
                return float(d)

        module_scores = {
            "stage1_dtw": _f("stage1"),
            "stage2_dtw": _f("stage2"),
            "completeness": _f("completeness"),
            "coordination": _f("coordination"),
            "knee_power": _f("knee_power"),
            "release_angle": _f("release_angle"),
            "height": _f("height"),
        }
        try:
            final = ScoringEngine.combine_scores(module_scores, Config.SCORE_WEIGHTS)
        except Exception:
            final = float(np.mean(list(module_scores.values())))
        return {
            "final_score": round(final, 2),
            "stage1_dtw": round(module_scores["stage1_dtw"], 2),
            "stage2_dtw": round(module_scores["stage2_dtw"], 2),
            "completeness": round(module_scores["completeness"], 2),
            "coordination": round(module_scores["coordination"], 2),
            "knee_power": round(module_scores["knee_power"], 2),
            "release_angle": round(module_scores["release_angle"], 2),
            "height": round(module_scores["height"], 2),
        }

    def _build_mem_shot_detail(self, shot, shot_id_str):
        """组装内存单投详情（scores/reports/ai 走内存，frames/pose/data 走磁盘）。"""
        sd = shot.get("save_data") or {}
        shot_dir = sd.get("shot_dir")
        session_dir = sd.get("session_dir")
        data_json = self._load_shot_data_json(shot_dir)
        return {
            "shot_id": shot_id_str,
            "shot_idx": shot.get("shot_idx"),
            "persisted": bool(shot_dir and os.path.isdir(shot_dir)),
            "start_time": shot.get("start_time_str"),
            "end_time": shot.get("end_time_str"),
            "duration": shot.get("duration_str"),
            "start_idx": shot.get("start_idx"),
            "release_idx": shot.get("release_idx"),
            "scores": shot.get("scores"),
            "reports": shot.get("reports"),
            "ai_report": shot.get("ai_report"),
            "frames": self._scan_shot_frames(shot_dir),
            "videos": self._scan_session_videos(session_dir),
            "pose": data_json.get("list_pose") if data_json else None,
            "data": data_json,
        }

    def _build_disk_shot_detail(self, shot_dir, shot_id_str, session_dir):
        """组装历史单投详情（全走磁盘 data.json，final_score 重算）。"""
        data_json = self._load_shot_data_json(shot_dir)
        scoring = (data_json or {}).get("scoring", {})
        return {
            "shot_id": shot_id_str,
            "persisted": True,
            "start_time": (data_json or {}).get("start_time"),
            "end_time": (data_json or {}).get("end_time"),
            "start_frame": (data_json or {}).get("start_frame"),
            "end_frame": (data_json or {}).get("end_frame"),
            "scores": self._scores_from_disk_scoring(scoring),
            "reports": None,  # 历史 data.json 不存模块报告文本
            "ai_report": scoring.get("ai_comment", "") if scoring else "",
            "frames": self._scan_shot_frames(shot_dir),
            "videos": self._scan_session_videos(session_dir),
            "pose": (data_json or {}).get("list_pose"),
            "data": data_json,
        }

    def _render_shot_sub(self, detail, sub):
        """按子资源名返回 detail 对应切片；sub=None 返回总览。"""
        if sub is None:
            return jsonify({"code": 200, "shot": detail})
        mapping = {
            "scores": detail.get("scores"),
            "reports": detail.get("reports"),
            "ai": detail.get("ai_report"),
            "frames": detail.get("frames"),
            "pose": detail.get("pose"),
            "data": detail.get("data"),
        }
        if sub not in mapping:
            raise HttpApiError("未知子资源: %s" % sub, errno=404)
        return jsonify({
            "code": 200,
            "shot_id": detail.get("shot_id"),
            sub: mapping[sub],
        })

    def _summarize_mem_shot(self, shot):
        """内存 shot -> 列表摘要（shot_id + final_score + 时间 + media 入口）。

        兼容 Qt 客户端旧解析：同时透出 shot_idx / scores（七项分）/ ai_report，
        避免 /result 拆分后 Qt 端 _show_result 按旧结构取不到字段、全显示 "-"。
        """
        shot_id = self._mem_shot_id(shot)
        sd = shot.get("save_data") or {}
        scores = shot.get("scores") or {}
        base = self._public_base()
        return {
            "shot_id": shot_id,
            # ── 兼容 Qt 旧字段（_show_result 依赖这三项）──────────────
            "shot_idx": shot.get("shot_idx"),
            "scores": scores,
            "ai_report": shot.get("ai_report"),
            # ── 摘要字段 ───────────────────────────────────────────────
            "final_score": scores.get("final_score"),
            "start_time": shot.get("start_time_str"),
            "end_time": shot.get("end_time_str"),
            "duration": shot.get("duration_str"),
            "media": {
                "detail_url": (base + "result/" + shot_id) if shot_id else None,
                "frames_url": (base + "result/" + shot_id + "/frames")
                              if shot_id else None,
                "videos": self._scan_session_videos(sd.get("session_dir")),
            },
        }

    def _mem_result_summary(self):
        """会话统计：总投数 + 平均/最高/最低 + 各维度均值。

        输出字段按标准顺序编排：code / total_shots / avg_score / max_score /
        min_score / dimension_avg。dimension_avg 固定按 7 个维度顺序输出
        （stage1_dtw / stage2_dtw / completeness / coordination / knee_power /
        release_angle / height），无数据的维度值为 None，保证结构稳定。
        """
        shots = self.inference.results
        dim_keys = ("stage1_dtw", "stage2_dtw", "completeness",
                    "coordination", "knee_power", "release_angle", "height")
        # 预置全部维度 key，保证输出字段顺序固定、缺失维度也不丢失字段
        dims = {k: [] for k in dim_keys}
        finals = []
        for shot in shots:
            s = shot.get("scores") or {}
            f = s.get("final_score")
            if f is not None:
                finals.append(float(f))
            for k in dim_keys:
                v = s.get(k)
                if v is not None:
                    dims[k].append(float(v))
        dimension_avg = {k: (round(sum(v) / len(v), 2) if v else None)
                         for k, v in dims.items()}
        return {
            "code": 200,
            "total_shots": len(shots),
            "avg_score": round(sum(finals) / len(finals), 2) if finals else None,
            "max_score": round(max(finals), 2) if finals else None,
            "min_score": round(min(finals), 2) if finals else None,
            "dimension_avg": dimension_avg,
        }

    def _scan_sessions(self, limit=200):
        """扫描 save_data 下所有历史会话，返回会话摘要列表（按时间倒序）。"""
        root = os.path.abspath(Config.SAVE_DATA_ROOT)
        sessions = []
        if not os.path.isdir(root):
            return sessions
        for date_name in sorted(os.listdir(root), reverse=True):
            date_dir = os.path.join(root, date_name)
            if not os.path.isdir(date_dir):
                continue
            for name in sorted(os.listdir(date_dir), reverse=True):
                sess_dir = os.path.join(date_dir, name)
                if not os.path.isdir(sess_dir):
                    continue
                sessions.append(
                    self._build_session_summary(date_name, name, sess_dir))
                if len(sessions) >= limit:
                    return sessions
        return sessions

    def _find_session_dir(self, name):
        """按会话目录名（不含日期前缀）在 save_data 下定位会话目录。"""
        root = os.path.abspath(Config.SAVE_DATA_ROOT)
        if not os.path.isdir(root):
            return None
        for date_name in os.listdir(root):
            date_dir = os.path.join(root, date_name)
            if not os.path.isdir(date_dir):
                continue
            cand = os.path.join(date_dir, name)
            if os.path.isdir(cand):
                return cand
        return None

    def _build_session_summary(self, date_name, name, sess_dir):
        """组装单个会话摘要：元数据 + 投篮编号列表 + 视频列表。"""
        meta = self._load_session_meta(sess_dir)
        shots = self._scan_session_shots(sess_dir)
        return {
            "date": date_name,
            "session_name": name,
            "user_id": meta.get("user_id") or "0000",
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "shot_count": len(shots),
            "shots": [{
                "shot_id": sid,
                "detail_url": self._public_base()
                              + "sessions/" + name + "/shots/" + sid,
            } for sid in shots],
            "videos": self._scan_session_videos(sess_dir),
        }

    def _scan_session_shots(self, sess_dir):
        """扫描会话 images/ 下所有投篮编号目录（三位），返回编号列表。"""
        images_dir = SaveDataLayout.images_dir(sess_dir)
        if not os.path.isdir(images_dir):
            return []
        shots = []
        for name in sorted(os.listdir(images_dir)):
            if re.fullmatch(r"\d{3}", name) and \
                    os.path.isdir(os.path.join(images_dir, name)):
                shots.append(name)
        return shots

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
            # 肩/肘/髋/膝关键点不可见时角度为 None，透传 None 给客户端（Qt 侧显示 '-'）
            def _f(v):
                return round(float(v), 2) if v is not None else None
            angles = {
                "shoulder": _f(shoulder),
                "elbow": _f(elbow),
                "hip": _f(hip),
                "knee": _f(knee),
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
