# -*- coding: utf-8 -*-
"""
HTTP 客户端封装（QT_Linux/http_client）
========================================
职责：封装与 RK3588 算法盒子 HTTP 服务的通信（连接/open/close/start/stop/record/
      frames/result/status）。所有网络异常在本层捕获并归类返回，不向 UI 层抛异常。

约定：
  - 每个方法返回 (ok: bool, data: dict|None, err: str)
  - ok=True 时 data 为解析后的 JSON（或说明字符串），err 为空
  - ok=False 时 data 为 None，err 为错误描述

依赖：requests
"""

import json
import logging

import requests

logger = logging.getLogger("qt_client")


class ApiClient:
    """RK3588 算法服务 HTTP 客户端（Linux 端默认连本机 127.0.0.1）。"""

    TIMEOUT = 10.0  # 单次请求超时（秒）；open/close 需等 MPP 解码器清理，适当放宽

    def __init__(self, host="127.0.0.1", port=8899):
        self.host = host
        self.port = int(port)

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def set_endpoint(self, host, port):
        self.host = host
        self.port = int(port)

    # ------------------------------------------------------------------
    # 底层请求
    # ------------------------------------------------------------------
    def _request(self, method, path, **kwargs):
        url = self.base_url + path
        kwargs.setdefault("timeout", self.TIMEOUT)
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.exceptions.ConnectTimeout:
            return False, None, f"连接超时（{self.TIMEOUT}s）: {url}"
        except requests.exceptions.ConnectionError as e:
            return False, None, f"连接失败（算法服务未启动或 IP/端口错误）: {e}"
        except requests.exceptions.ReadTimeout:
            return False, None, f"读取超时（{self.TIMEOUT}s）: {url}"
        except Exception as e:
            return False, None, f"请求异常（{type(e).__name__}: {e}）"

        if resp.status_code >= 400:
            return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        try:
            data = resp.json()
        except ValueError:
            # 非 JSON 响应（如图片字节），原样返回文本
            return True, {"raw": resp.content}, ""
        return True, data, ""

    def _post(self, path, **kwargs):
        ok, data, err = self._request("POST", path, **kwargs)
        if not ok:
            return False, None, err
        code = data.get("code", -1)
        msg = data.get("msg", "")
        if code != 200:
            return False, data, f"{msg or '未知错误'} (code={code})"
        return True, data, ""

    def _get(self, path, params=None):
        return self._request("GET", path, params=params)

    # ------------------------------------------------------------------
    # 业务接口
    # ------------------------------------------------------------------
    def health(self):
        ok, data, err = self._get("/health")
        if not ok:
            return False, None, err
        return data.get("code") == 200, data, err

    def open_camera(self):
        """POST /open 打开摄像头。"""
        return self._post("/open")

    def close_camera(self):
        """POST /close 关闭摄像头。"""
        return self._post("/close")

    def start_motion(self, user_id=None):
        """POST /start 开始运动（录像 + 识别）。

        user_id：可选，用户 ID。会随会话保存到 RK3588 端 save_data 的会话元数据中。
        """
        if user_id:
            user_id = str(user_id).strip()
            if user_id:
                return self._post("/start", json={"user_id": user_id})
        return self._post("/start")

    def stop_motion(self):
        """POST /stop 停止运动。"""
        return self._post("/stop")

    def record(self):
        """POST /record 开始录像。"""
        return self._post("/record")

    def record_stop(self):
        """POST /record/stop 停止录像并保存。"""
        return self._post("/record/stop")

    def pause(self):
        """POST /pause 暂停。"""
        return self._post("/pause")

    def get_status(self):
        """GET /status 通道状态。"""
        return self._get("/status")

    def get_result(self):
        """GET /result 分析结果（综合得分/各项得分/AI 评语）。"""
        return self._get("/result")

    def get_frames(self, n=1, with_meta=True):
        """GET /frames 单帧图片（附带关键点/骨架/角度元数据）。"""
        return self._get("/frames", params={"n": n, "meta": 1 if with_meta else 0})

    def get_frames_raw(self):
        """GET /frames/raw：获取 RK3588 硬解码的「原始 BGR 帧」+ AI 元数据。

        返回 (ok, raw_bytes, meta_dict, err)：
          - ok=True 时 raw_bytes 为 BGR 裸字节流，meta_dict 含 width/height 及
            meta(关键点/角度/框)，供 Qt 端用 QImage(Format_BGR888) 直接重建，无需 OpenCV。
          - ok=False 时 raw_bytes=None, meta_dict=None。
        """
        url = self.base_url + "/frames/raw"
        try:
            resp = requests.get(url, timeout=self.TIMEOUT)
        except requests.exceptions.ConnectTimeout:
            return False, None, None, f"连接超时（{self.TIMEOUT}s）: {url}"
        except requests.exceptions.ConnectionError as e:
            return False, None, None, f"连接失败: {e}"
        except requests.exceptions.ReadTimeout:
            return False, None, None, f"读取超时（{self.TIMEOUT}s）: {url}"
        except Exception as e:
            return False, None, None, f"请求异常（{type(e).__name__}: {e}）"

        if resp.status_code >= 400:
            return False, None, None, f"HTTP {resp.status_code}: {resp.text[:200]}"

        raw = resp.content
        try:
            w = int(resp.headers.get("X-Frame-Width", 0))
            h = int(resp.headers.get("X-Frame-Height", 0))
        except (TypeError, ValueError):
            w = h = 0
        meta = {}
        meta_json = resp.headers.get("X-Frame-Meta", "")
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except ValueError as e:
                logger.warning("X-Frame-Meta 解析失败: %s", e)
        if not raw or w <= 0 or h <= 0:
            return False, None, None, "空帧或尺寸非法"
        meta["width"] = w
        meta["height"] = h
        return True, raw, meta, ""
