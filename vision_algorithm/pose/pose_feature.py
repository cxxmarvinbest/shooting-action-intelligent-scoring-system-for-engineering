# -*- coding: utf-8 -*-
"""
姿态特征提取模块（pose_estimate/pose_feature）
================================================
职责：从姿态关键点中提取投篮动作特征（关节角度、髋部纵坐标、手腕/肩部纵坐标、像素身高等）。

对外暴露：
  - SKELETON_CONNECTIONS —— COCO 骨架连线（供可视化复用）
  - calculate_angle —— 三点夹角
  - extract_pose_features —— 单帧关键点 -> 特征 dict

依赖：numpy
"""

import numpy as np


# ── 骨架连接（COCO 关键点连线，用于可视化）──
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]


def calculate_angle(a, b, c):
    """计算以 b 为顶点、a/c 为两边的夹角（度）"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_pose_features(kpts, player_box):
    """
    从单帧关键点提取动作特征。

    参数：
        kpts        —— (17,2) ndarray，输入帧坐标系，不可见点坐标为 0
        player_box —— 主球员框 (x1,y1,x2,y2)，可为 None

    返回：dict，键为 angles / side_str / wrist_y / shoulder_y / player_h / hip_y，
          缺失项为 None。与原 frame_metrics 结构完全一致。
    """
    features = {
        'angles': None, 'side_str': None, 'wrist_x': None, 'wrist_y': None,
        'shoulder_y': None, 'player_h': None, 'hip_y': None,
        'ankle_angle': None,
    }
    if kpts is None:
        return features

    abs_kpts = np.zeros_like(kpts)
    for i in range(len(kpts)):
        if kpts[i][0] > 0:
            abs_kpts[i] = [kpts[i][0], kpts[i][1]]

    r_s, r_e, r_w = abs_kpts[6], abs_kpts[8], abs_kpts[10]
    l_s, l_e, l_w = abs_kpts[5], abs_kpts[7], abs_kpts[9]

    # 区分左右侧（右侧可见优先）
    if r_w[0] > 0 and r_s[0] > 0:
        s, e, w, h, k, a = r_s, r_e, r_w, abs_kpts[12], abs_kpts[14], abs_kpts[16]
        side_str = "Right"
    else:
        s, e, w, h, k, a = l_s, l_e, l_w, abs_kpts[11], abs_kpts[13], abs_kpts[15]
        side_str = "Left"

    if s[0] > 0 and h[0] > 0:
        shoulder = calculate_angle(h, s, e) if e[0] > 0 else 160.0
        elbow = calculate_angle(s, e, w) if (e[0] > 0 and w[0] > 0) else 180.0
        hip = calculate_angle(s, h, k) if k[0] > 0 else 180.0
        knee = calculate_angle(h, k, a) if (k[0] > 0 and a[0] > 0) else 180.0

        features['angles'] = [shoulder, elbow, hip, knee]
        features['side_str'] = side_str
        features['wrist_x'] = w[0]
        features['wrist_y'] = w[1]
        features['shoulder_y'] = s[1]
        if player_box is not None:
            features['player_h'] = player_box[3] - player_box[1]

        # 踝角（近似）：小腿（膝→踝）与竖直向下方向的夹角，用于 Qt 客户端展示
        if k[0] > 0 and a[0] > 0:
            dx = a[0] - k[0]
            dy = a[1] - k[1]  # 图像坐标系 y 向下为正
            shin_len = (dx * dx + dy * dy) ** 0.5
            if shin_len > 1e-3:
                cos_v = dy / shin_len  # 小腿方向与竖直向下(0,1)的夹角余弦
                features['ankle_angle'] = float(np.degrees(
                    np.arccos(np.clip(cos_v, -1.0, 1.0))))

    # 髋部中心纵坐标（左右髋均值）
    hips = [kpts[11], kpts[12]]
    valid_hips_y = [pt[1] for pt in hips if pt[0] > 0]
    if valid_hips_y:
        features['hip_y'] = sum(valid_hips_y) / len(valid_hips_y)

    return features
