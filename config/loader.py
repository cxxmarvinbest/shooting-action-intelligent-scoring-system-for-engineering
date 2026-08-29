# -*- coding: utf-8 -*-
"""
配置加载器（config/loader）
=============================
把 config/ 目录下的多个 YAML 配置文件合并为一个扁平的 Config 对象，
对外提供属性式访问（Config.DET_RKNN_PATH / Config.SCORE_WEIGHTS ...）。

设计要点：
  1. 一个 YAML 一个领域（settings/paths/model/inference/...），改某类参数只动对应文件；
  2. 加载时把所有 YAML 的顶层 key 合并成一个 dict，key 保持大写、与旧 config.py 的
     类属性名一致，从而「from config import Config」与「Config.XXX」无需改动；
  3. 环境变量优先级最高：LQ_* 环境变量可覆盖对应字段（部署机免改 YAML），
     覆盖时按 YAML 默认值的类型自动做 int/float/bool/json 转换；
  4. 路径类字段（*_PATH / *_DIR / *_CACHE_PATH）若为相对路径，自动 join PROJECT_ROOT。
"""

import json
import os

import yaml

# config/ 目录与项目根目录
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

# 配置字段 -> 环境变量名 映射（仅列需要支持环境变量覆盖的字段，与原 config.py 一致）
ENV_OVERRIDES = {
    "DET_RKNN_PATH": "LQ_DET_RKNN",
    "POSE_RKNN_PATH": "LQ_POSE_RKNN",
    "POSE_KPT_CONF_THRES": "LQ_POSE_KPT_CONF_THRES",
    "NPU_CORE_MASK": "LQ_NPU_CORE_MASK",
    "STANDARD_VIDEO_DIR": "LQ_STD_DIR",
    "OUTPUT_DIR": "LQ_OUT_DIR",
    "STANDARD_CACHE_PATH": "LQ_STANDARD_CACHE",
    "CROP_X_OFFSET": "LQ_CROP_X_OFFSET",
    "FRAME_STRIDE": "LQ_FRAME_STRIDE",
    "RING_PRECACHE_FRAMES": "LQ_RING_PRECACHE",
    "RING_MAX_FRAMES": "LQ_RING_MAX",
    "HOLD_DEBOUNCE_FRAMES": "LQ_HOLD_DEBOUNCE",
    "HOLD_TIMEOUT_FRAMES": "LQ_HOLD_TIMEOUT",
    "LOOKBACK_WINDOW_FRAMES": "LQ_LOOKBACK_WINDOW",
    "RELEASE_TRIGGER_WINDOW": "LQ_RELEASE_WINDOW",
    "BALL_WRIST_DIST_RATIO": "LQ_BALL_WRIST_DIST",
    "KNEE_SQUAT_THRESHOLD": "LQ_KNEE_SQUAT_THR",
    "KNEE_STAND_MIN": "LQ_KNEE_STAND_MIN",
    "KNEE_STABLE_FRAMES": "LQ_KNEE_STABLE",
    "MIN_SHOT_FRAMES": "LQ_MIN_SHOT_FRAMES",
    "PHASE_DTW_RATIO": "LQ_PHASE_DTW_RATIO",
    "SCORE_WEIGHTS": "LQ_SCORE_WEIGHTS",
    "OFFLINE_MODE": "LQ_OFFLINE",
    "SHOW_PREVIEW": "LQ_SHOW_PREVIEW",
    "REALTIME_STATUS_INTERVAL": "LQ_RT_STATUS_INTERVAL",
    "RECORD_ROTATE_SEC": "LQ_RECORD_ROTATE",
}

# 需要 join PROJECT_ROOT 的相对路径字段
PATH_KEYS = {
    "WEIGHTS_DIR", "LOG_DIR", "RECORD_DIR",
    "DET_RKNN_PATH", "POSE_RKNN_PATH",
    "STANDARD_VIDEO_DIR", "OUTPUT_DIR", "STANDARD_CACHE_PATH",
}


def _coerce(value, reference):
    """按 reference 的类型把字符串 value 转换为对应类型。"""
    if isinstance(reference, bool):
        return bool(int(value))
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, dict):
        return json.loads(value) if isinstance(value, str) else value
    return value  # str 原样返回


def _load_all():
    """加载 config/*.yaml 并合并为扁平 dict。"""
    cfg = {}
    for fname in sorted(os.listdir(CONFIG_DIR)):
        if not fname.endswith(".yaml") and not fname.endswith(".yml"):
            continue
        path = os.path.join(CONFIG_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        cfg.update(data)

    # 环境变量覆盖 + 按默认值类型转换
    for key, env_name in ENV_OVERRIDES.items():
        env_val = os.environ.get(env_name)
        if key in cfg and env_val not in (None, ""):
            cfg[key] = _coerce(env_val, cfg[key])

    # 相对路径字段 join PROJECT_ROOT（已是绝对路径则原样保留）
    for key in PATH_KEYS:
        if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])

    return cfg


class _Config:
    """只读配置对象：属性访问 + 支持 .get()，禁止运行时篡改配置。"""

    def __init__(self, data):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"未知配置项: {name}")

    def __setattr__(self, name, value):
        raise AttributeError("Config 为只读对象，请通过 YAML 或 LQ_* 环境变量修改配置")

    def get(self, name, default=None):
        return self._data.get(name, default)

    def as_dict(self):
        return dict(self._data)


Config = _Config(_load_all())
