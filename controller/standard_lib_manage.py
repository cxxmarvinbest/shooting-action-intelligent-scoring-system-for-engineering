# -*- coding: utf-8 -*-
"""
标准库预加载管理（controller/standard_lib_manage）
====================================================
职责：进程启动时一次性加载 RKNN 检测/姿态模型 + 标准视频库特征（幂等），
      供离线评分与实时识别共用，避免重复 load_models / 重复跑标准库推理。

对外暴露：StandardLibManage
依赖：config / vision_algorithm.pipeline / vision_algorithm.standard
"""

import logging
import threading

from config import Config
from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer
from vision_algorithm.standard.standard_library import (
    StandardLibrary, list_standard_videos)

logger = logging.getLogger("basketball_scoring")


class StandardLibManage:
    """模型 + 标准视频库的预加载器（幂等，仅加载一次）。"""

    def __init__(self):
        self.analyzer = None
        self.std_cache = {}          # champ1 / champ2 / avg_std_height
        self.std_video_count = 0
        self._lock = threading.Lock()

    def ensure_models(self):
        """加载 RKNN 模型 + 标准视频库特征（幂等）。"""
        if self.analyzer is not None:
            return
        with self._lock:
            if self.analyzer is not None:
                return
            logger.info("加载 RKNN 检测/姿态模型 ...")
            self.analyzer = VideoAnalyzer()
            self.analyzer.load_models()

            logger.info("模型加载完成，预加载标准视频库特征 ...")
            std_lib = StandardLibrary(self.analyzer)
            std_videos = list_standard_videos(Config.STANDARD_VIDEO_DIR)
            std_lib.build(std_videos, cache_path=Config.STANDARD_CACHE_PATH)
            self.std_cache['champ1'] = std_lib.champ1
            self.std_cache['champ2'] = std_lib.champ2
            self.std_cache['avg_std_height'] = std_lib.avg_std_height
            self.std_video_count = len(std_videos)
            logger.info("标准库加载完成（%d 个标准视频）", self.std_video_count)
