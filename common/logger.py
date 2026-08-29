# -*- coding: utf-8 -*-
"""
统一日志模块（common/logger）
===============================
提供 setup_logger：控制台 + 文件双输出，避免重复注册 handler。
原 main.py / http_server.py 各有一套重复的日志初始化逻辑，此处统一。
"""

import logging
import os
import sys
import time

_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def setup_logger(log_dir, name="basketball_scoring", level=logging.INFO):
    """初始化日志（控制台 + 文件），返回日志文件路径。

    :param log_dir: 日志目录（不存在会自动创建）
    :param name:    logger 名称
    :param level:   日志级别
    :return:        日志文件完整路径
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, time.strftime("scoring_%Y%m%d_%H%M%S.log"))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 避免作为模块被多次调用时重复注册 handler

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_FMT)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_FMT)
    logger.addHandler(fh)

    return log_file


def get_logger(name="basketball_scoring"):
    """获取日志器（未 setup 时返回默认 logger）。"""
    return logging.getLogger(name)
