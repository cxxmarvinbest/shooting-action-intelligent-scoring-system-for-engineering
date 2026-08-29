# -*- coding: utf-8 -*-
"""
统一异常定义模块（common/exceptions）
=====================================
按「异常来源」分类，便于日志定位与 HTTP 状态码映射。各业务层捕获底层异常后，
统一抛/记录本模块的异常，使错误可被精确定位到「哪一层、哪一类、什么原因」。

分类：
  - HttpApiError         HTTP 接口层异常（路径错误 / 非法参数 / 视频损坏）
  - RtspStreamError      RTSP 拉流异常（断流 / 视频损坏 / 解码失败）
  - RknnInferenceError   RKNN 推理异常（推理输入异常 / NPU 资源异常）
  - VideoSplitError      视频切分异常（数组长度不一致 / 输入数据异常）
  - ScoringError         打分异常（数组长度不一致 / 输入数据异常）
  - ConfigError          配置异常

每个异常带 errno（错误码）与 kind（子类目），HTTP 层据此映射状态码与提示文案。

依赖：无（纯标准库，可被任意模块安全 import）
"""


class BaseAlgoError(Exception):
    """算法系统统一异常基类。"""

    kind = "unknown"
    errno = -1

    def __init__(self, message, *, kind=None, errno=None, cause=None):
        self.message = str(message)
        if kind is not None:
            self.kind = kind
        if errno is not None:
            self.errno = errno
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        if self.cause is not None:
            return f"[{self.kind}] {self.message}（原因: {self.cause}）"
        return f"[{self.kind}] {self.message}"


class HttpApiError(BaseAlgoError):
    """HTTP 接口层异常。"""

    kind = "http"


class RtspStreamError(BaseAlgoError):
    """RTSP 拉流异常。"""

    kind = "rtsp"


class RknnInferenceError(BaseAlgoError):
    """RKNN 推理异常。"""

    kind = "rknn"


class VideoSplitError(BaseAlgoError):
    """视频切分异常。"""

    kind = "video_split"


class ScoringError(BaseAlgoError):
    """打分异常。"""

    kind = "scoring"


class ConfigError(BaseAlgoError):
    """配置异常。"""

    kind = "config"


def classify_exception(exc, default_kind="unknown"):
    """把一个未知异常归类为 BaseAlgoError（若不是 BaseAlgoError 则包装）。

    便于在统一捕获点把任意异常转换为可识别分类，避免在业务层散落大量
    isinstance 判断。返回 BaseAlgoError 实例（原样返回已归类异常）。
    """
    if isinstance(exc, BaseAlgoError):
        return exc
    return BaseAlgoError(str(exc), kind=default_kind, cause=exc)
