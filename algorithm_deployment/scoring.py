# -*- coding: utf-8 -*-
"""
评分模块（scoring）
===================
职责：对检测结果（frame_metrics 逐帧特征、动作角度序列、出手高度等）进行量化评分。
包含：
  1. 核心环节技术完整度评估（下蹲蓄力 / 蹬伸发力 / 出手释放）
  2. 屈膝发力与爆发性评估（下蹲幅度 + 蹬伸角速度）
  3. 动力链协同与发力节奏评估（五节点达峰时序）
  4. 出手角度评估（出手瞬间小臂对地夹角）
  5. 出手高度评估（相对出手高度差值）
  6. DTW 距离换算与冠军样本选择、标准视频比对

对外只暴露：ScoringEngine（全部为无状态静态方法）
依赖：numpy / fastdtw / scipy.spatial.distance（不依赖检测、LLM、UI 模块）
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class ScoringEngine:
    """评分计算引擎（纯计算，无状态，可独立单元测试）"""

    # ============================================================
    # 核心环节技术完整度
    # ============================================================
    @staticmethod
    def compute_completeness(frame_metrics, fps=30.0):
        """
        动作完成度专项评估模块
        结合髋关节纵向坐标的[显著下压]判定下蹲蓄力、[反向回弹]判定蹬伸发力
        """
        total_frames = len(frame_metrics)
        duration = total_frames / fps if fps > 0 else 0.0

        # 1. 提取纵向重心特征 (髋关节中心 Y 坐标) 及 球员像素高度
        hip_ys = [m['hip_y'] for m in frame_metrics if m.get('hip_y') is not None]
        player_heights = [m['player_h'] for m in frame_metrics if m.get('player_h') is not None]
        avg_h = np.mean(player_heights) if player_heights else 300.0

        # 2. 提取髋关节和膝关节角度序列
        hip_angles = [m['angles'][2] for m in frame_metrics if m.get('angles') is not None]
        knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]

        has_squat = False
        has_extension = False
        has_release = False

        # 内部一维滑动平均平滑函数，滤除追踪抖动
        def smooth(data, window=5):
            if len(data) < window:
                return np.array(data)
            # 边缘用端点值延拓，避免 'same' 模式在序列两端除数不足导致的幅值衰减
            pad = window // 2
            padded = np.pad(np.asarray(data, dtype=float), pad, mode='edge')
            return np.convolve(padded, np.ones(window) / window, mode='valid')

        # 执行复合形态学判定
        if len(hip_ys) > 5:
            smoothed_hips = smooth(hip_ys)

            # 找到全视频中髋关节纵向的最低点（即图像坐标系中 Y 的最大值）
            max_y_idx = np.argmax(smoothed_hips)
            max_y_val = smoothed_hips[max_y_idx]

            # 计算从视频开始到最低点期间，身体曾达到的最高位置（Y 的最小值，通常是准备站立姿态）
            min_y_before = np.min(smoothed_hips[:max_y_idx + 1]) if max_y_idx > 0 else smoothed_hips[0]

            # 计算从最低点到视频结束期间，身体向上的反向腾跃/展直高度（Y 的最小值）
            min_y_after = np.min(smoothed_hips[max_y_idx:]) if max_y_idx < len(smoothed_hips) - 1 else max_y_val

            # 归一化：计算重心下蹲和向上反弹占总身高的比例
            drop_ratio = (max_y_val - min_y_before) / avg_h
            rise_ratio = (max_y_val - min_y_after) / avg_h

            # 获取全过程中的关节极限弯曲角度
            min_hip_angle = np.min(hip_angles) if hip_angles else 180.0
            min_knee_angle = np.min(knee_angles) if knee_angles else 180.0

            # -- 🎯 核心判别条件 1：下蹲蓄力环节 --
            is_coordinate_dropped = drop_ratio > 0.04
            is_direction_changed = (drop_ratio > 0.02) and (rise_ratio > 0.02)
            is_angle_flexed = (min_hip_angle < 160.0) or (min_knee_angle < 152.0)

            if is_coordinate_dropped or is_direction_changed or is_angle_flexed:
                has_squat = True

            # -- 🎯 核心判别条件 2：蹬伸发力环节 --
            post_min_knee = knee_angles[max_y_idx:] if knee_angles and max_y_idx < len(knee_angles) else []
            if post_min_knee and (np.max(post_min_knee) - np.min(knee_angles) > 15) and np.max(post_min_knee) > 155:
                has_extension = True
            elif rise_ratio > 0.05:
                has_extension = True
        else:
            if knee_angles and np.min(knee_angles) < 145:
                has_squat = True
            if knee_angles and np.max(knee_angles) - np.min(knee_angles) > 20:
                has_extension = True

        # -- 🎯 核心判别条件 3：出手释放环节 --
        for m in frame_metrics:
            if m.get('angles') is not None and m.get('wrist_y') is not None and m.get('shoulder_y') is not None:
                if m['wrist_y'] < m['shoulder_y'] and m['angles'][1] > 140:
                    has_release = True
                    break

        # 3. 生成可读性报告
        stages_status = []
        missing_details = []

        if has_squat:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 下蹲蓄力环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 下蹲蓄力环节</b></font>")
            missing_details.append("❌ 缺乏有效的下蹲蓄力（髋关节重心无明显下压，或缺乏‘下蹲-蹬伸’的方向相反坐标衔接）")

        if has_extension:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 蹬伸发力环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 蹬伸发力环节</b></font>")
            missing_details.append("❌ 缺乏蹬伸环节（重心最低点后未见身体及膝、髋关节有效向上延展）")

        if has_release:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 出手释放环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 出手释放环节</b></font>")
            missing_details.append("❌ 缺乏出手环节（未见手腕举起过肩或肘关节未能有效伸直推球）")

        completed_count = sum([has_squat, has_extension, has_release])
        score = (completed_count / 3.0) * 100.0
        conclusion = "<font color='#A6E3A1'>🎉 恭喜！投篮核心技术环节完整，动作链衔接良好。</font>" if score == 100.0 else "<br>".join(
            missing_details)

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center' width='40%'><b>指标项</b></th>
                <th align='center' width='60%'><b>数据与判别结果</b></th>
            </tr>
            <tr>
                <td align='center'>⏱️ 动作完成时间</td>
                <td align='center'><b>{duration:.2f} 秒</b> (共计 {total_frames} 帧)</td>
            </tr>
            <tr>
                <td align='center'>🔍 核心技术环节检测</td>
                <td align='left' style='line-height:22px;'>
                    · {stages_status[0]}<br>
                    · {stages_status[1]}<br>
                    · {stages_status[2]}
                </td>
            </tr>
            <tr>
                <td align='center'>💡 针对性改进意见</td>
                <td align='left' style='color:#F9E2AF; line-height:20px;'>{conclusion}</td>
            </tr>
        </table>
        """
        return score, html_report

    # ============================================================
    # 屈膝发力与爆发性
    # ============================================================
    @staticmethod
    def compute_knee_power(frame_metrics, fps=30.0):
        """屈膝发力模块：结合下蹲幅度与蹬伸角速度评估爆发力"""
        knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]
        if len(knee_angles) < 5:
            return 0.0, "<div align='center'>膝关节数据不足，无法评估屈膝发力</div>"

        def smooth(data, window=5):
            if len(data) < window:
                return np.array(data)
            # 边缘用端点值延拓，避免 'same' 模式在序列两端除数不足导致的幅值衰减
            pad = window // 2
            padded = np.pad(np.asarray(data, dtype=float), pad, mode='edge')
            return np.convolve(padded, np.ones(window) / window, mode='valid')

        smoothed_knee = smooth(knee_angles)
        min_knee = np.min(smoothed_knee)
        max_knee = np.max(smoothed_knee)
        amplitude = max_knee - min_knee

        velocities = np.diff(smoothed_knee) * fps
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0

        amp_score = 100.0 - abs(amplitude - 75.0) * 1.5
        amp_score = max(0.0, min(100.0, amp_score))

        vel_score = (max_velocity / 350.0) * 100.0
        vel_score = max(0.0, min(100.0, vel_score))

        total_score = (amp_score * 0.5) + (vel_score * 0.5)

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center'><b>评价指标</b></th>
                <th align='center'><b>实测数据</b></th>
                <th align='center'><b>单项得分</b></th>
            </tr>
            <tr><td align='center'>最低下蹲角度</td><td align='center'>{min_knee:.1f}°</td><td align='center'>-</td></tr>
            <tr><td align='center'>膝关节屈伸幅度</td><td align='center'>{amplitude:.1f}°</td><td align='center'>{amp_score:.1f}</td></tr>
            <tr><td align='center'>最大蹬伸角速度</td><td align='center'>{max_velocity:.1f}°/s</td><td align='center'>{vel_score:.1f}</td></tr>
        </table>
        """
        return total_score, html_report

    # ============================================================
    # 动力链协同与发力节奏
    # ============================================================
    @staticmethod
    def compute_coordination(frame_metrics):
        """动力链协同模块：五节点达峰时序一致性评估"""
        if len(frame_metrics) < 10:
            return 0.0, "数据量过少，无法分析动力链"

        hip_angles = [m['angles'][2] if m.get('angles') is not None else 180 for m in frame_metrics]
        knee_angles = [m['angles'][3] if m.get('angles') is not None else 180 for m in frame_metrics]
        shoulder_angles = [m['angles'][0] if m.get('angles') is not None else 180 for m in frame_metrics]
        elbow_angles = [m['angles'][1] if m.get('angles') is not None else 180 for m in frame_metrics]

        wrist_ys = []
        for m in frame_metrics:
            w_y = m.get('wrist_y', None)
            if w_y is not None:
                wrist_ys.append(w_y)
            else:
                wrist_ys.append(wrist_ys[-1] if wrist_ys else 9999)

        def smooth(data, window=5):
            if len(data) < window:
                return np.array(data)
            # 边缘用端点值延拓，避免 'same' 模式在序列两端除数不足导致的幅值衰减
            pad = window // 2
            padded = np.pad(np.asarray(data, dtype=float), pad, mode='edge')
            return np.convolve(padded, np.ones(window) / window, mode='valid')

        t_hip = int(np.argmax(smooth(hip_angles)))
        t_knee = int(np.argmax(smooth(knee_angles)))
        t_shoulder = int(np.argmax(smooth(shoulder_angles)))
        t_elbow = int(np.argmax(smooth(elbow_angles)))
        t_release = int(np.argmin(smooth(wrist_ys)))

        peaks = {
            "髋部伸展": t_hip,
            "膝部蹬伸": t_knee,
            "肩部发力": t_shoulder,
            "肘部传递": t_elbow,
            "手腕释放": t_release
        }

        score = 100.0
        lags = [
            ("膝-髋 (下肢)", t_knee - t_hip),
            ("肩-膝 (躯干)", t_shoulder - t_knee),
            ("肘-肩 (上肢)", t_elbow - t_shoulder),
            ("腕-肘 (末端)", t_release - t_elbow)
        ]

        for name, lag in lags:
            if lag < 0:
                score += lag * 3

        score = max(0.0, min(100.0, score))

        t_start = min(peaks.values())
        t_end = max(peaks.values())
        total_time = t_end - t_start if t_end > t_start else 1

        html_rows = ""
        sorted_peaks = sorted(peaks.items(), key=lambda x: x[1])
        for name, t in sorted_peaks:
            rel_pct = ((t - t_start) / total_time) * 100
            html_rows += f"<tr><td align='center'>{name}</td><td align='center'>第 {t} 帧</td><td align='center'>{rel_pct:.1f}%</td></tr>"

        report = f"""
                <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
                    <tr style='color:#89B4FA;'>
                        <th align='center'><b>动力链环节</b></th>
                        <th align='center'><b>达峰节点</b></th>
                        <th align='center'><b>相对总耗时比例</b></th>
                    </tr>
                    {html_rows}
                </table>
                """
        return score, report

    # ============================================================
    # 出手角度
    # ============================================================
    @staticmethod
    def compute_release_angle(frame_metrics):
        """出手角度模块：出手瞬间小臂对地夹角"""
        valid_frames = [m for m in frame_metrics if m.get('wrist_y') is not None and m.get('kpts') is not None]
        if not valid_frames:
            return 0.0, "<div align='center'>缺失手腕或关键点数据，无法评估出手角度</div>"

        release_frame = min(valid_frames, key=lambda x: x['wrist_y'])
        kpts = release_frame['kpts']
        side = release_frame.get('side_str', 'Right')

        if side == 'Right':
            e_idx, w_idx = 8, 10
        else:
            e_idx, w_idx = 7, 9

        elbow = kpts[e_idx]
        wrist = kpts[w_idx]

        if elbow[0] == 0 or wrist[0] == 0:
            return 0.0, "<div align='center'>出手瞬间手臂关键点被遮挡，无法计算角度</div>"

        dx = abs(wrist[0] - elbow[0])
        dy = elbow[1] - wrist[1]

        if dx == 0 and dy == 0:
            angle = 0.0
        else:
            angle = float(np.degrees(np.arctan2(dy, dx)))

        score = 100.0 - abs(angle - 50.0) * 2.5
        score = max(0.0, min(100.0, score))

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center'><b>评价指标</b></th>
                <th align='center'><b>实测角度</b></th>
                <th align='center'><b>单项得分</b></th>
            </tr>
            <tr>
                <td align='center'>出手小臂对地夹角</td>
                <td align='center'>{angle:.1f}°</td>
                <td align='center'>{score:.1f}</td>
            </tr>
        </table>
        """
        return score, html_report

    # ============================================================
    # DTW 距离换算
    # ============================================================
    @staticmethod
    def compute_dtw_score(true_avg_degree):
        """将 DTW 平均距离换算为 0~100 分数"""
        if true_avg_degree <= 10.0:
            return 100.0
        elif true_avg_degree >= 55.0:
            return 20.0
        else:
            return 100 - (true_avg_degree - 10) * (80 / 45)

    # ============================================================
    # 标准视频比对：冠军样本选择 + DTW 距离计算
    # ============================================================
    @staticmethod
    def select_champion(std_seqs):
        """从标准视频序列库中选出与其他样本平均 DTW 距离最小的"冠军样本"作为比对基准"""
        min_dist, champ = float('inf'), std_seqs[0]
        for i, sq_a in enumerate(std_seqs):
            tot = sum(fastdtw(sq_a, sq_b, dist=euclidean)[0] / max(len(sq_a), len(sq_b))
                      for j, sq_b in enumerate(std_seqs) if i != j)
            if len(std_seqs) > 1 and tot / (len(std_seqs) - 1) < min_dist:
                min_dist, champ = tot / (len(std_seqs) - 1), sq_a
        return champ

    @staticmethod
    def compute_dtw_distance(champ, test_seq):
        """计算测试序列与冠军样本的 DTW 距离，并换算为分数；返回 (deg, score)"""
        d, p = fastdtw(champ, test_seq, dist=euclidean)
        deg = (d / len(p)) / np.sqrt(4)
        score = ScoringEngine.compute_dtw_score(deg)
        return deg, score

    # ============================================================
    # 出手高度
    # ============================================================
    @staticmethod
    def compute_height_score(test_rel_h, avg_std_height):
        """出手高度评分：测试相对高度与标准参考相对高度的差值映射为分数"""
        height_diff = abs(test_rel_h - avg_std_height)
        return max(0.0, min(100.0, 100.0 - height_diff * 150.0))
