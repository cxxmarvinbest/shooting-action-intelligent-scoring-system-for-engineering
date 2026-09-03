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


def extract_pose_features(kpts, player_box, kpt_conf=None):
    """
    从单帧关键点提取动作特征。

    参数：
        kpts        —— (17,2) ndarray，输入帧坐标系，不可见点坐标为 0
        player_box —— 主球员框 (x1,y1,x2,y2)，可为 None
        kpt_conf    —— (17,) 各关键点置信度，可为 None；用于构造 visible_mask

    返回：dict，键为 angles / side_str / wrist_x / wrist_y / shoulder_y / player_h /
          hip_y / ankle_angle / visible_mask。
          关键点缺失时角度置 None（不再用 160/180 兜底），由上层 FSM 迟滞层容错。
    """
    features = {
        'angles': None, 'side_str': None, 'wrist_x': None, 'wrist_y': None,
        'shoulder_y': None, 'player_h': None, 'hip_y': None,
        'ankle_angle': None, 'visible_mask': None,
    }
    if kpts is None:
        return features

    kpts = np.asarray(kpts, dtype=np.float32)
    n = kpts.shape[0]

    # 可见性掩码：坐标非 0 为可见；若提供 kpt_conf，conf<=0 的点强制判不可见。
    # 说明：pose_model 已把低置信关键点坐标置 0，此处 conf 仅作二次兜底，
    #       同时把可见性透传出去供 FSM / 阶段 2 样本分析使用。
    coord_visible = (kpts[:, 0] > 0) | (kpts[:, 1] > 0)
    if kpt_conf is not None:
        conf = np.asarray(kpt_conf, dtype=np.float32).reshape(-1)
        visible = coord_visible & (conf > 0.0)
    else:
        visible = coord_visible
    features['visible_mask'] = visible.tolist()

    def _vis(i):
        return bool(visible[i]) if i < n else False

    # 左右侧关键点索引（COCO 17 点）：右侧优先（右肩 6 + 右腕 10 可见则用右侧）
    if _vis(6) and _vis(10):
        s_i, e_i, w_i, h_i, k_i, a_i = 6, 8, 10, 12, 14, 16
        side_str = "Right"
    else:
        s_i, e_i, w_i, h_i, k_i, a_i = 5, 7, 9, 11, 13, 15
        side_str = "Left"

    s, e, w = kpts[s_i], kpts[e_i], kpts[w_i]
    h, k, a = kpts[h_i], kpts[k_i], kpts[a_i]

    # 角度：三点任一不可见 -> 该角度置 None（不再兜底 160/180）
    if _vis(s_i) and _vis(h_i):
        shoulder = calculate_angle(h, s, e) if (_vis(s_i) and _vis(e_i) and _vis(h_i)) else None
        elbow = calculate_angle(s, e, w) if (_vis(s_i) and _vis(e_i) and _vis(w_i)) else None
        hip = calculate_angle(s, h, k) if (_vis(s_i) and _vis(h_i) and _vis(k_i)) else None
        knee = calculate_angle(h, k, a) if (_vis(h_i) and _vis(k_i) and _vis(a_i)) else None

        features['angles'] = [shoulder, elbow, hip, knee]
        features['side_str'] = side_str
        features['wrist_x'] = float(w[0]) if _vis(w_i) else None
        features['wrist_y'] = float(w[1]) if _vis(w_i) else None
        features['shoulder_y'] = float(s[1]) if _vis(s_i) else None
        if player_box is not None:
            features['player_h'] = player_box[3] - player_box[1]

        # 踝角（近似）：小腿（膝→踝）与竖直向下方向的夹角，用于 Qt 客户端展示
        if _vis(k_i) and _vis(a_i):
            dx = a[0] - k[0]
            dy = a[1] - k[1]  # 图像坐标系 y 向下为正
            shin_len = (dx * dx + dy * dy) ** 0.5
            if shin_len > 1e-3:
                cos_v = dy / shin_len  # 小腿方向与竖直向下(0,1)的夹角余弦
                features['ankle_angle'] = float(np.degrees(
                    np.arccos(np.clip(cos_v, -1.0, 1.0))))

    # 髋部中心纵坐标（左右髋均值）
    valid_hips_y = [float(kpts[i][1]) for i in (11, 12) if _vis(i)]
    if valid_hips_y:
        features['hip_y'] = sum(valid_hips_y) / len(valid_hips_y)

    return features
