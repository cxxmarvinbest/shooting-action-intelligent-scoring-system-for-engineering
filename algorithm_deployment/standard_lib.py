# -*- coding: utf-8 -*-
"""
标准视频库预加载模块（standard_lib）
====================================
职责：把「标准视频库 → 冠军样本角度序列 + 平均出手高度」的推理与选样过程
      只执行一次，并缓存到磁盘（.npz）。后续每次评分直接读缓存，避免
      对 29 个标准视频重复跑 NPU 推理。

缓存失效：缓存文件会记录生成时的 FRAME_STRIDE，读取时与当前配置比对，
          不一致则自动重新生成。

对外暴露：StandardLibrary
依赖：numpy / config / scoring / pipeline（VideoAnalyzer）
"""

import logging
import os

import numpy as np

from config import Config
from scoring import ScoringEngine

logger = logging.getLogger("basketball_scoring")


class StandardLibrary:
    """标准视频库预加载器：只跑一次推理，缓存冠军样本与平均出手高度"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.champ1 = None        # 阶段1（准备-下蹲）冠军角度序列 (K1, 4)
        self.champ2 = None        # 阶段2（蹬伸-出手）冠军角度序列 (K2, 4)
        self.avg_std_height = 0.5

    def build(self, standard_videos, cache_path=None):
        """
        构建标准库特征（优先读缓存）。

        参数：
            standard_videos —— 标准视频路径列表
            cache_path      —— .npz 缓存文件路径；None 则不落盘
        """
        # 1) 优先读磁盘缓存（重启进程也不用重算）
        if cache_path and os.path.exists(cache_path):
            if self._load_cache(cache_path):
                return

        # 2) 对标准库跑一次推理（仅一次）
        std_seqs1, std_seqs2, std_heights = [], [], []
        for i, path in enumerate(standard_videos):
            s1, s2, _, _, rel_h, _ = self.analyzer.process_video(
                path.strip(), save_visuals=False)
            if s1 is not None and len(s1) > 2:
                std_seqs1.append(s1)
            if s2 is not None and len(s2) > 2:
                std_seqs2.append(s2)
            if rel_h > 0:
                std_heights.append(rel_h)
            logger.info("标准视频 %d/%d 完成: %s",
                        i + 1, len(standard_videos), os.path.basename(path.strip()))

        self.avg_std_height = sum(std_heights) / len(std_heights) if std_heights else 0.5
        if not std_seqs1 or not std_seqs2:
            raise ValueError("标准视频库解析失败，无法提取两段动作特征！")

        # 3) 选冠军样本（与其他样本平均 DTW 距离最小的"最居中"样本）
        self.champ1 = ScoringEngine.select_champion(std_seqs1)
        self.champ2 = ScoringEngine.select_champion(std_seqs2)
        logger.info("冠军样本选择完成（champ1=%d 帧, champ2=%d 帧）",
                    len(self.champ1), len(self.champ2))

        # 4) 写缓存
        if cache_path:
            self._save_cache(cache_path)

    def _load_cache(self, cache_path):
        """读缓存；成功且未失效返回 True，否则返回 False 触发重新生成"""
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_stride = int(data["frame_stride"]) if "frame_stride" in data else None
            if cached_stride != int(Config.FRAME_STRIDE):
                logger.warning("缓存与当前 FRAME_STRIDE 不一致（缓存=%s, 当前=%s），重新生成",
                               cached_stride, Config.FRAME_STRIDE)
                return False
            self.champ1 = data["champ1"]
            self.champ2 = data["champ2"]
            self.avg_std_height = float(data["avg_std_height"])
            logger.info("已从缓存加载标准库特征: %s（champ1=%d 帧, champ2=%d 帧）",
                        cache_path, len(self.champ1), len(self.champ2))
            return True
        except Exception as e:
            logger.warning("缓存读取失败（%s），重新生成标准库特征", e)
            return False

    def _save_cache(self, cache_path):
        try:
            parent = os.path.dirname(cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez(cache_path,
                     champ1=self.champ1,
                     champ2=self.champ2,
                     avg_std_height=self.avg_std_height,
                     frame_stride=int(Config.FRAME_STRIDE))
            logger.info("标准库特征已缓存: %s", cache_path)
        except Exception as e:
            logger.warning("缓存写入失败（%s），本次仍可继续评分", e)
