# -*- coding: utf-8 -*-
"""
目标筛选模块（tracker/target_selector）
========================================
职责：从检测/姿态结果中筛选出"主球员"与"有效篮球"目标。
  1. select_main_player_box —— 从检测结果中按面积选出主球员框（cls==player）
  2. select_main_player     —— 从姿态结果中按面积选出主球员（兼容旧调用）
  3. filter_balls           —— 从检测结果中按长宽比/尺寸/位置约束筛选篮球

依赖：config
"""

from config import Config


class TargetSelector:
    """球员/篮球目标筛选器（无状态，可复用）"""

    @staticmethod
    def select_main_player_box(dets, player_cls_id):
        """
        从检测结果中选出面积最大的 player 框（cls == player_cls_id）。

        参数：dets —— RKNNDetModel 检测结果列表（坐标为检测画布/工作帧坐标系均可）
              player_cls_id —— player 的类别 id（由调用方从 Config.DET_PLAYER_CLS_ID 显式传入）
        返回：player 框 (x1,y1,x2,y2)；无目标时返回 None
        """
        candidates = [d for d in dets if d['cls'] == player_cls_id]
        if not candidates:
            return None
        best = max(candidates, key=lambda d: (d['box'][2] - d['box'][0])
                   * (d['box'][3] - d['box'][1]))
        return best['box']

    @staticmethod
    def select_main_player(poses):
        """
        从姿态结果中选出面积最大的人体框作为主球员。

        参数：poses —— RKNNPoseModel.detect 返回的列表
        返回：(player_box, kpts, kpt_conf)
              player_box 为 (x1,y1,x2,y2)，kpts 为 (17,2)，kpt_conf 为 (17,)
              无目标时返回 (None, None, None)
        """
        if not poses:
            return None, None, None
        best = max(poses, key=lambda p: (p['box'][2] - p['box'][0])
                   * (p['box'][3] - p['box'][1]))
        return best['box'], best['kpts'], best.get('kpt_conf')

    @staticmethod
    def filter_balls(dets, player_box, frame_w):
        """
        从检测结果中筛选有效篮球目标（cls == Config.DET_BALL_CLS_ID）。

        约束（与原版一致）：
          - 长宽比 < BALL_MAX_ASPECT
          - 宽度 < 画面宽 * BALL_MAX_W_RATIO
          - 球心纵坐标不显著低于球员框底边（排除地面干扰）

        参数：dets —— RKNNDetModel.detect 返回的列表
              player_box —— 主球员框 (x1,y1,x2,y2)，可为 None
              frame_w —— 画面宽度
        返回：[{'box': (x1,y1,x2,y2), 'conf': float}] 列表
               （按置信度降序后至多保留 1 个；conf 供阶段 0 持球/出手判定加权）
        """
        ball_cls_id = Config.get("DET_BALL_CLS_ID", 1)
        balls = []
        for d in dets:
            if d['cls'] != ball_cls_id:
                continue
            bx1, by1, bx2, by2 = d['box']
            w, h = bx2 - bx1, by2 - by1
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio >= Config.BALL_MAX_ASPECT:
                continue
            if w >= frame_w * Config.BALL_MAX_W_RATIO:
                continue
            if player_box is not None:
                px1, py1, px2, py2 = player_box
                ball_cy = (by1 + by2) / 2
                if ball_cy > py2 + 10:
                    continue
            # 保留置信度：阶段 0 需要 ball_conf 做持球/出手判定加权，不能只留 box
            balls.append({'box': d['box'], 'conf': float(d.get('conf', 0.0))})

        # det 结果已按置信度降序，取首个即最优目标
        return balls[:1]
