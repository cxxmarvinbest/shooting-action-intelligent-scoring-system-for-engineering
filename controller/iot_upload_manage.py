# -*- coding: utf-8 -*-
"""
后端 IoT 上传客户端（controller/iot_upload_manage）
====================================================
职责：设备鉴权（get_device_token）+ 运动记录上传（addAlgorithm）。

设计要点：
  1. token 缓存 + 过期重取：get_device_token 成功后缓存 token 与过期时间戳，
     每次上传前 _auth_token() 判断是否临近过期（提前 IOT_TOKEN_CACHE_ADVANCE 秒），
     过期则重新获取；并发由 _token_lock 保护（双重检查，避免并发重复请求）。
  2. 同步调用：上传在 /stop 时按需触发，不继承 ThreadBase（无长驻网络线程）。
  3. 开关：IOT_UPLOAD_ENABLED=false 时 pipeline 不装配本模块，零影响。
  4. requests 未安装 / 请求失败时降级返回 None（仅记日志，不中断主流程）。

对外暴露：IotUploadManage
依赖：requests / config
"""

import logging
import threading
import time

from config import Config

logger = logging.getLogger("basketball_scoring")

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    _REQUESTS_AVAILABLE = False


class IotUploadManage:
    """后端 IoT 上传客户端：token 缓存 + 运动记录上传。"""

    def __init__(self):
        self.get_token_url = Config.get("IOT_GET_TOKEN_URL", "")
        self.add_algorithm_url = Config.get("IOT_ADD_ALGORITHM_URL", "")
        self.device_identification = Config.get("IOT_DEVICE_IDENTIFICATION", "")
        self.order_id = Config.get("IOT_ORDER_ID", "")
        self.cache_advance = int(Config.get("IOT_TOKEN_CACHE_ADVANCE", 300))
        self.timeout = float(Config.get("IOT_UPLOAD_TIMEOUT", 30))
        # token 缓存：_token / _expire_ts(epoch 秒)；0 表示未知/未获取
        self._token = None
        self._expire_ts = 0.0
        self._token_lock = threading.Lock()

    # ------------------------------------------------------------------
    # token 缓存 + 过期重取
    # ------------------------------------------------------------------
    def _token_valid(self):
        """token 是否有效（存在且未临近过期）。"""
        return bool(self._token) and time.time() < self._expire_ts - self.cache_advance

    def get_device_token(self, force=False):
        """获取设备令牌（带缓存；过期或 force=True 时重取）。

        返回 token 字符串；失败返回 None。
        """
        if not _REQUESTS_AVAILABLE:
            logger.warning("[IotUpload] requests 未安装，无法获取设备令牌")
            return None
        if not force and self._token_valid():
            return self._token
        with self._token_lock:
            # 双重检查：拿到锁后可能已被其他线程刷新
            if not force and self._token_valid():
                return self._token
            try:
                resp = requests.post(
                    self.get_token_url,
                    json={"device_identification": self.device_identification},
                    timeout=self.timeout,
                    proxies={"http": None, "https": None},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error("[IotUpload] 获取设备令牌失败（%s: %s）",
                             type(e).__name__, e)
                return None
            if not isinstance(data, dict) or data.get("code") != 0 \
                    or not data.get("data"):
                logger.error("[IotUpload] 获取设备令牌返回异常: %r", data)
                return None
            token = (data["data"] or {}).get("token")
            expire_ms = (data["data"] or {}).get("expire_timestamp")
            if not token:
                logger.error("[IotUpload] 设备令牌为空")
                return None
            self._token = token
            try:
                self._expire_ts = float(expire_ms) / 1000.0 if expire_ms else 0.0
            except (TypeError, ValueError):
                self._expire_ts = 0.0
            logger.info("[IotUpload] 设备令牌获取成功（过期时间戳=%s）", expire_ms)
            return token

    def _auth_token(self):
        """确保 token 有效（临近过期则重取），返回 token 或 None。"""
        if self._token_valid():
            return self._token
        return self.get_device_token()

    # ------------------------------------------------------------------
    # 运动记录上传
    # ------------------------------------------------------------------
    def add_algorithm(self, payload):
        """上传运动记录到 addAlgorithm 接口。

        payload：已含 token 的完整请求体 dict。
        返回后端返回的 data 对象（通常含 exercise_record_id）；失败返回 None。
        """
        if not _REQUESTS_AVAILABLE:
            logger.warning("[IotUpload] requests 未安装，无法上传运动记录")
            return None
        if not self.add_algorithm_url:
            logger.error("[IotUpload] 未配置 IOT_ADD_ALGORITHM_URL")
            return None
        try:
            resp = requests.post(
                self.add_algorithm_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json; charset=utf-8"},
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("[IotUpload] 上传运动记录失败（%s: %s）",
                         type(e).__name__, e)
            return None
        if not isinstance(data, dict) or data.get("code") != 0:
            logger.error("[IotUpload] 上传运动记录返回异常: %r", data)
            return None
        result = data.get("data")
        logger.info("[IotUpload] 运动记录上传成功: %r", result)
        return result

    def upload_session(self, payload):
        """上传一次运动会话记录：先确保 token，再上传。

        返回后端 data 对象（含 exercise_record_id）或 None。
        """
        token = self._auth_token()
        if not token:
            return None
        payload = dict(payload or {})
        payload["token"] = token
        return self.add_algorithm(payload)
