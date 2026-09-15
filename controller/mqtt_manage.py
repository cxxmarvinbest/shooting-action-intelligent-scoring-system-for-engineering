# -*- coding: utf-8 -*-
"""
MQTT 实时事件推送模块（controller/mqtt_manage）
================================================
职责：作为 MQTT client 连接外部 EMQX broker，向上游 APP 推送轻量事件，并
      订阅下行指令 topic 做基础控制（MQTT→HTTP 桥）。

事件协议（上行，topic = MQTT_TOPIC_TX，payload 全 JSON 轻量，不含图片/视频）：
  {"type":"status",    "state":"ready|running|stopped|error", ...}   # 状态变化
  {"type":"shot_done", "shot_id","user_id","final_score","scores",
   "start_time","end_time","duration","detail_url", ...}             # 每投一次
  {"type":"cmd_ack",   "cmd","code","msg", ...}                      # 指令应答
  {"type":"query_result","shots":[...]}                              # query_shot_detail 应答

下行指令（topic = MQTT_TOPIC_RX）：
  {"cmd":"start","user_id":"U123"} / {"cmd":"stop"} / {"cmd":"pause"}
  {"cmd":"status"} / {"cmd":"query_shot_detail","shot_id":"001"}

设计要点：
  1. 自动重连：paho reconnect_delay_set 指数退避 + loop_forever 内建重连；
  2. 开关：MQTT_ENABLED=false 时 pipeline 不装配本模块，零影响；
  3. 线程安全：publish 由 paho 内部锁保护，可被 inference/http 线程并发调用；
  4. paho-mqtt 未安装时降级为 no-op（不 import 失败、不启动网络线程）。

对外暴露：MqttManage（继承 ThreadBase，loop_forever 跑在子线程）
依赖：paho-mqtt / config
"""

import json
import logging

from config import Config
from common.thread_base import ThreadBase

logger = logging.getLogger("basketball_scoring")

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    mqtt = None
    _PAHO_AVAILABLE = False


class MqttManage(ThreadBase):
    """MQTT 客户端：推送 status / shot_done，订阅下行指令。"""

    def __init__(self, command_handler=None):
        super().__init__(name="MqttManage")
        self.host = Config.get("MQTT_HOST", "127.0.0.1")
        self.port = int(Config.get("MQTT_PORT", 1883))
        self.client_id = Config.get("MQTT_CLIENT_ID", "basketball_scoring")
        self.username = Config.get("MQTT_USERNAME", "") or ""
        self.password = Config.get("MQTT_PASSWORD", "") or ""
        self.topic_tx = Config.get("MQTT_TOPIC_TX", "/SS/BA/DMT/001/AI/TX")
        self.topic_rx = Config.get("MQTT_TOPIC_RX", "/SS/BA/DMT/001/AI/RX")
        self.qos = int(Config.get("MQTT_QOS", 1))
        self.keepalive = int(Config.get("MQTT_KEEPALIVE", 60))
        # 下行指令回调：handler(cmd: str, data: dict)，跑在 MQTT 网络线程
        self.command_handler = command_handler
        self._client = None
        self._connected = False

    @property
    def is_connected(self):
        return self._connected and self._client is not None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, join_timeout=None):
        if not _PAHO_AVAILABLE:
            logger.warning("[MqttManage] paho-mqtt 未安装，MQTT 推送已禁用")
            return
        super().start(join_timeout)

    def stop(self):
        super().stop()
        client = self._client
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass

    def _run(self):
        if not _PAHO_AVAILABLE:
            return
        client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        self._client = client
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        if self.username:
            client.username_pw_set(self.username, self.password)
        client.reconnect_delay_set(
            min_delay=int(Config.get("MQTT_RECONNECT_MIN_DELAY", 1)),
            max_delay=int(Config.get("MQTT_RECONNECT_MAX_DELAY", 30)))
        try:
            client.connect(self.host, self.port, keepalive=self.keepalive)
        except Exception as e:
            logger.error("[MqttManage] 连接 broker 失败（%s: %s），本会话不启用 MQTT",
                         type(e).__name__, e)
            self._client = None
            return
        logger.info("[MqttManage] 连接 broker %s:%d ...", self.host, self.port)
        client.loop_forever()  # 阻塞，直到 disconnect() 被调用
        self._connected = False
        self._client = None
        logger.info("[MqttManage] MQTT 已断开")

    # ------------------------------------------------------------------
    # paho 回调
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("[MqttManage] 已连接 broker %s:%d", self.host, self.port)
            try:
                client.subscribe(self.topic_rx, qos=self.qos)
                logger.info("[MqttManage] 已订阅下行指令: %s", self.topic_rx)
            except Exception as e:
                logger.error("[MqttManage] 订阅下行指令失败: %s", e)
            # 连接成功即推一次在线状态
            self.publish_status("ready")
        else:
            self._connected = False
            logger.warning("[MqttManage] 连接被拒绝 rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("[MqttManage] 意外断开 rc=%d，等待自动重连...", rc)

    def _on_message(self, client, userdata, msg):
        try:
            text = (msg.payload or b"").decode("utf-8", errors="ignore").strip()
            data = json.loads(text) if text else {}
        except Exception as e:
            logger.warning("[MqttManage] 下行指令解析失败: %s", e)
            return
        if not isinstance(data, dict):
            data = {}
        cmd = data.get("cmd")
        if not cmd:
            logger.warning("[MqttManage] 下行指令缺少 cmd 字段: %r", data)
            return
        logger.info("[MqttManage] 收到下行指令 cmd=%s", cmd)
        if self.command_handler is not None:
            try:
                self.command_handler(cmd, data)
            except Exception as e:
                logger.error("[MqttManage] 指令处理失败 cmd=%s（%s: %s）",
                             cmd, type(e).__name__, e)

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    def _publish(self, payload_dict):
        if not self.is_connected:
            return False
        try:
            payload = json.dumps(payload_dict, ensure_ascii=False)
            self._client.publish(self.topic_tx, payload, qos=self.qos)
            return True
        except Exception as e:
            logger.warning("[MqttManage] 发布失败（%s: %s）", type(e).__name__, e)
            return False

    def publish_status(self, state, extra=None):
        """推送程序状态变化（state: ready/running/stopped/error 等）。"""
        payload = {"type": "status", "state": state}
        if extra:
            payload.update(extra)
        return self._publish(payload)

    def publish_shot_done(self, summary):
        """推送投篮完成事件（summary 为轻量摘要 dict）。"""
        payload = {"type": "shot_done"}
        payload.update(summary or {})
        return self._publish(payload)

    def publish_cmd_ack(self, cmd, code, msg, extra=None):
        """推送指令应答。"""
        payload = {"type": "cmd_ack", "cmd": cmd, "code": code, "msg": msg}
        if extra:
            payload.update(extra)
        return self._publish(payload)

    def publish_query_result(self, shots):
        """推送 query_shot_detail 应答（轻量摘要列表）。"""
        payload = {"type": "query_result", "shots": shots or []}
        return self._publish(payload)
