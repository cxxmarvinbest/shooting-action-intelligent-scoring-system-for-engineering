# -*- coding: utf-8 -*-
"""
评分模块（scoring）
===================
职责：对检测结果（frame_metrics 逐帧特征、动作角度序列、出手高度等）进行量化评分。
包含：
  1. 核心环节技术完整度评估（下蹲蓄力 / 蹬伸发力 / 出手释放）
  2. 屈髋屈膝发力与爆发性评估（髋/膝屈曲幅度 + 蹬伸角速度，含噪声伪峰过滤）
  3. 动力链协同与发力节奏评估（下蹲最低点之后的伸展峰值 + 各关节启动时序）
  4. 出手角度评估（出手瞬间小臂对地夹角）
  5. 出手高度评估（相对出手高度差值）

对外只暴露：ScoringEngine（全部为无状态静态方法）
依赖：numpy / fastdtw / scipy.spatial.distance（不依赖检测、LLM、UI 模块）
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from common.exceptions import ScoringError


class ScoringEngine:
    """评分计算引擎（纯计算，无状态，可独立单元测试）"""

    # 动力链同步时间窗口（帧）：相邻关节达峰/启动时差在此窗口内视为同步。
    # 实测经验值取 2~4 帧，默认 3；可按视频帧率微调。
    CHAIN_WINDOW_FRAMES = 3

    # ============================================================
    # 内部通用工具
    # ============================================================
    @staticmethod
    def _validate_angle_sequence(seq, name="序列"):
        """校验角度序列为合法的二维数值数组（行>=2、列一致），否则抛 ScoringError。

        用于捕获「数组长度不一致 / 输入数据异常」这类打分异常。
        """
        if seq is None:
            raise ScoringError(f"{name}为 None（输入数据异常）", kind="scoring")
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2:
            raise ScoringError(
                f"{name}维度非法：应为 2 维，实际 {arr.ndim} 维（数组长度不一致）",
                kind="scoring")
        if arr.shape[0] < 2:
            raise ScoringError(
                f"{name}样本数不足：应 >= 2，实际 {arr.shape[0]}（数组长度不一致）",
                kind="scoring")
        if arr.shape[1] == 0:
            raise ScoringError(
                f"{name}特征维度为 0（输入数据异常）", kind="scoring")
        return arr

    @staticmethod
    def _smooth(data, window=5):
        """一维滑动平均平滑，滤除追踪抖动；长度不足窗口时原样返回。"""
        if len(data) < window:
            return np.asarray(data, dtype=float)
        pad = window // 2
        padded = np.pad(np.asarray(data, dtype=float), pad, mode='edge')
        return np.convolve(padded, np.ones(window) / window, mode='valid')

    @staticmethod
    def _aligned_angles(frame_metrics, idx, default=180.0):
        """按帧序提取指定关节角度序列（与 frame_metrics 逐帧对齐，缺帧用 default 填充）。

        idx: 0=肩, 1=肘, 2=髋, 3=膝
        """
        return np.array([
            m['angles'][idx] if (m.get('angles') is not None) else default
            for m in frame_metrics
        ], dtype=float)

    @staticmethod
    def _aligned_hip_y(frame_metrics):
        """按帧序提取髋部中心纵坐标（图像 Y 越大越低），缺帧线性插值补全。"""
        n = len(frame_metrics)
        arr = np.full(n, np.nan)
        for i, m in enumerate(frame_metrics):
            if m.get('hip_y') is not None:
                arr[i] = m['hip_y']
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            return arr
        idx = np.where(valid)[0]
        return np.interp(np.arange(n), idx, arr[valid])

    @staticmethod
    def _aligned_wrist_y(frame_metrics):
        """按帧序提取手腕纵坐标（图像 Y 越小越高），缺帧线性插值补全。"""
        n = len(frame_metrics)
        arr = np.full(n, np.nan)
        for i, m in enumerate(frame_metrics):
            if m.get('wrist_y') is not None:
                arr[i] = m['wrist_y']
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            return np.full(n, 100000.0)
        idx = np.where(valid)[0]
        return np.interp(np.arange(n), idx, arr[valid])

    @staticmethod
    def _find_squat_bottom(frame_metrics, smooth_win=5):
        """定位下蹲最低点帧号。

        优先用髋部中心纵坐标（Y 最大=最低），数据不足时回退到膝关节最小角度。
        返回值为 frame_metrics 的全局帧号。
        """
        n = len(frame_metrics)
        if n == 0:
            return 0
        hips = ScoringEngine._aligned_hip_y(frame_metrics)
        if np.isfinite(hips).sum() >= 3:
            sm = ScoringEngine._smooth(hips, smooth_win)
            return int(np.argmax(sm))
        knee = ScoringEngine._aligned_angles(frame_metrics, 3)
        sm = ScoringEngine._smooth(knee, smooth_win)
        return int(np.argmin(sm))

    @staticmethod
    def _find_valid_peaks(signal, min_height=None, min_prominence=None, min_distance=1):
        """在平滑后的一维信号中寻找有效局部极大值，并过滤噪声伪峰。

        判据（依次过滤）：
          1. 局部极大值（严格 >= 左右邻点，端点除外）
          2. 峰值高度 >= min_height（若给定）
          3. 突出度 >= min_prominence（若给定）：峰高相对左右两侧谷底的高度差
          4. 相邻有效峰间距 >= min_distance（贪心保留更高峰）

        返回：有效峰索引列表（升序）。
        """
        signal = np.asarray(signal, dtype=float)
        n = len(signal)
        if n < 3:
            return []

        peak_mask = np.zeros(n, dtype=bool)
        for i in range(1, n - 1):
            if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
                peak_mask[i] = True
        cand = np.where(peak_mask)[0]
        if cand.size == 0:
            return []

        # 先按峰值高度降序，便于贪心保留高峰
        order = sorted(cand.tolist(), key=lambda i: -signal[i])

        selected = []
        for i in order:
            if min_height is not None and signal[i] < min_height:
                continue
            if min_distance and any(abs(i - s) < min_distance for s in selected):
                continue
            if min_prominence is not None:
                # 向左/右扩展到局部下降尽头，找到两侧谷底，突出度=峰高-两侧谷底最大值
                left = i
                while left - 1 >= 0 and signal[left - 1] <= signal[left]:
                    left -= 1
                right = i
                while right + 1 < n and signal[right + 1] <= signal[right]:
                    right += 1
                base = max(signal[left], signal[right])
                if signal[i] - base < min_prominence:
                    continue
            selected.append(i)

        return sorted(selected)

    @staticmethod
    def _extension_onset_peak(signal, bottom, window, mode="max"):
        """在下蹲最低点之后的蹬伸阶段，计算单关节的「启动时序」与「伸展峰值」帧号。

        参数：
            signal —— 全序列（角度：伸展=增大；wrist_y：释放=减小）
            bottom —— 下蹲最低点帧号
            window —— 峰值最小间距（帧），用于合并相邻抖动伪峰
            mode   —— "max"：峰值取局部极大（各关节角度）；"min"：峰值取局部极小（手腕释放）

        返回：(onset, peak) 全局帧号。
        """
        seg = signal[bottom:]
        n = len(seg)
        if n < 2:
            return bottom, bottom

        lo = float(np.min(seg))
        hi = float(np.max(seg))
        span = hi - lo

        # 启动时序：以「从极值点变化 20% 幅度」作为开始伸展/释放的标志，
        # 避免把追踪噪声当成启动点。
        if span < 1e-3:
            onset = bottom
        elif mode == "max":
            thr = lo + 0.2 * span
            cross = np.where(seg >= thr)[0]
            onset = bottom + int(cross[0]) if cross.size else bottom
        else:
            thr = hi - 0.2 * span
            cross = np.where(seg <= thr)[0]
            onset = bottom + int(cross[0]) if cross.size else bottom

        # 峰值：平滑后找有效峰（过滤噪声伪峰），取幅度最大者
        sm = ScoringEngine._smooth(signal, 5)
        seg_sm = sm[bottom:]
        prominence = max(3.0, span * 0.15)
        if mode == "max":
            peaks = ScoringEngine._find_valid_peaks(
                seg_sm, min_prominence=prominence, min_distance=window)
            if peaks:
                best = int(peaks[np.argmax([seg_sm[p] for p in peaks])])
            else:
                best = int(np.argmax(seg_sm))
        else:
            # 谷值：对负信号找峰
            peaks = ScoringEngine._find_valid_peaks(
                -seg_sm, min_prominence=prominence, min_distance=window)
            if peaks:
                best = int(peaks[np.argmin([seg_sm[p] for p in peaks])])
            else:
                best = int(np.argmin(seg_sm))

        return onset, bottom + best

    # ============================================================
    # 核心环节技术完整度
    # ============================================================
    @staticmethod
    def compute_completeness(frame_metrics, fps=30.0):
        """
        动作完成度专项评估模块
        结合髋关节纵向坐标的[显著下压]判定下蹲蓄力、[反向回弹]判定蹬伸发力；
        蹬伸发力进一步用「蹬伸结束膝关节角度」判断是否充分蹬直。
        """
        total_frames = len(frame_metrics)
        duration = total_frames / fps if fps > 0 else 0.0

        player_heights = [m['player_h'] for m in frame_metrics if m.get('player_h') is not None]
        avg_h = np.mean(player_heights) if player_heights else 300.0

        hip = ScoringEngine._aligned_angles(frame_metrics, 2)
        knee = ScoringEngine._aligned_angles(frame_metrics, 3)
        sm_hip = ScoringEngine._smooth(hip, 5)
        sm_knee = ScoringEngine._smooth(knee, 5)
        hips = ScoringEngine._aligned_hip_y(frame_metrics)

        has_squat = False
        has_extension = False
        has_release = False
        end_knee_angle = None
        fully_extended = False

        bottom = ScoringEngine._find_squat_bottom(frame_metrics)

        # 出手释放帧：手腕举过肩且肘伸直推球（复用给蹬伸结束判定）
        release_idx = None
        for i, m in enumerate(frame_metrics):
            if (m.get('angles') is not None and m.get('wrist_y') is not None
                    and m.get('shoulder_y') is not None):
                if m['wrist_y'] < m['shoulder_y'] and m['angles'][1] > 140:
                    release_idx = i
                    break
        if release_idx is not None:
            has_release = True

        if len(frame_metrics) > 5:
            # 髋部纵向重心下压/反弹比例
            if np.isfinite(hips).sum() >= 3:
                sm_hips = ScoringEngine._smooth(hips, 5)
                max_y_idx = int(np.argmax(sm_hips))
                max_y_val = float(sm_hips[max_y_idx])
                min_y_before = float(np.min(sm_hips[:max_y_idx + 1])) if max_y_idx > 0 else float(sm_hips[0])
                min_y_after = float(np.min(sm_hips[max_y_idx:])) if max_y_idx < len(sm_hips) - 1 else max_y_val
                drop_ratio = (max_y_val - min_y_before) / avg_h
                rise_ratio = (max_y_val - min_y_after) / avg_h
            else:
                drop_ratio = rise_ratio = 0.0

            min_hip_angle = float(np.min(sm_hip)) if len(sm_hip) else 180.0
            min_knee_angle = float(np.min(sm_knee)) if len(sm_knee) else 180.0

            # -- 🎯 核心判别条件 1：下蹲蓄力环节 --
            is_coordinate_dropped = drop_ratio > 0.04
            is_direction_changed = (drop_ratio > 0.02) and (rise_ratio > 0.02)
            is_angle_flexed = (min_hip_angle < 160.0) or (min_knee_angle < 152.0)

            if is_coordinate_dropped or is_direction_changed or is_angle_flexed:
                has_squat = True

            # -- 🎯 核心判别条件 2：蹬伸发力环节（含充分蹬直判定）--
            post_knee = sm_knee[bottom:]
            if len(post_knee) > 0:
                max_post_knee = float(np.max(post_knee))
                ext_amplitude = max_post_knee - min_knee_angle

                # 蹬伸结束膝关节角度：取蹬伸阶段膝关节伸展峰值（有效峰，过滤噪声伪峰），
                # 向后取少量帧均值作为“蹬伸结束”时的膝关节角度，判断是否充分蹬直。
                ext_peaks = ScoringEngine._find_valid_peaks(
                    post_knee, min_prominence=5.0, min_distance=3)
                if ext_peaks:
                    peak_idx = int(ext_peaks[np.argmax([post_knee[p] for p in ext_peaks])])
                    end_knee_angle = float(np.mean(
                        post_knee[peak_idx:min(len(post_knee), peak_idx + 3)]))
                else:
                    end_knee_angle = max_post_knee

                # 充分蹬直：蹬伸结束时膝关节角度接近伸直（165° 以上，可调）
                fully_extended = end_knee_angle >= 165.0
                # 蹬伸环节完成 = 存在有效伸展幅度 且 蹬伸结束时膝关节充分蹬直
                has_extension = (ext_amplitude > 15.0) and fully_extended
            else:
                # 膝关节数据不可用时，退化为重心反弹比例判定
                has_extension = rise_ratio > 0.05
        else:
            if len(knee) and float(np.min(knee)) < 145:
                has_squat = True
            if len(knee) and float(np.max(knee)) - float(np.min(knee)) > 20:
                has_extension = True

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
            missing_details.append("❌ 缺乏蹬伸环节（重心最低点后未见身体及膝、髋关节有效向上延展，或蹬伸结束时膝关节未充分蹬直）")

        if has_release:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 出手释放环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 出手释放环节</b></font>")
            missing_details.append("❌ 缺乏出手环节（未见手腕举起过肩或肘关节未能有效伸直推球）")

        completed_count = sum([has_squat, has_extension, has_release])
        score = (completed_count / 3.0) * 100.0
        conclusion = "<font color='#A6E3A1'>🎉 恭喜！投篮核心技术环节完整，动作链衔接良好。</font>" if score == 100.0 else "<br>".join(
            missing_details)

        knee_end_html = "-"
        if end_knee_angle is not None:
            knee_end_html = f"{end_knee_angle:.1f}°（{'✅ 已充分蹬直' if fully_extended else '⚠️ 未充分蹬直'}）"

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
                <td align='center'>🦵 蹬伸结束膝关节角度</td>
                <td align='center'>{knee_end_html}</td>
            </tr>
            <tr>
                <td align='center'>💡 针对性改进意见</td>
                <td align='left' style='color:#F9E2AF; line-height:20px;'>{conclusion}</td>
            </tr>
        </table>
        """
        return score, html_report

    # ============================================================
    # 屈髋屈膝发力与爆发性
    # ============================================================
    @staticmethod
    def compute_knee_power(frame_metrics, fps=30.0):
        """屈髋屈膝发力与爆发性：综合髋/膝屈曲幅度与蹬伸角速度评估下肢爆发力。

        只在下蹲最低点之后的蹬伸阶段评估角速度，并对幅度峰值/角速度峰值做有效性过滤，
        剔除追踪抖动造成的噪声伪峰。
        """
        hip = ScoringEngine._aligned_angles(frame_metrics, 2)
        knee = ScoringEngine._aligned_angles(frame_metrics, 3)
        if len(hip) < 5 or len(knee) < 5:
            return 0.0, "<div align='center'>髋/膝关节数据不足，无法评估下肢发力</div>"

        bottom = ScoringEngine._find_squat_bottom(frame_metrics)

        # (名称, 序列, 理想屈伸幅度°, 理想最大蹬伸角速度°/s)
        joints = [
            ("髋关节(屈髋)", hip, 65.0, 300.0),
            ("膝关节(屈膝)", knee, 75.0, 350.0),
        ]

        html_rows = ""
        total = 0.0
        for name, series, ideal_amp, ideal_vel in joints:
            sm = ScoringEngine._smooth(series, 5)
            min_angle = float(np.min(sm[:bottom + 1])) if bottom + 1 <= len(sm) else float(np.min(sm))

            # 伸展幅度：只取蹬伸阶段（最低点之后）的有效伸展峰值，过滤噪声伪峰
            post = sm[bottom:]
            if len(post) > 0:
                amp_peaks = ScoringEngine._find_valid_peaks(
                    post, min_prominence=max(3.0, (float(np.max(post)) - min_angle) * 0.15),
                    min_distance=3)
                if amp_peaks:
                    post_max = float(np.max([post[p] for p in amp_peaks]))
                else:
                    post_max = float(np.max(post))
                amplitude = post_max - min_angle
            else:
                amplitude = 0.0

            # 蹬伸角速度：只看下蹲最低点之后的正向伸展速度，并做峰值有效性过滤
            vel = np.diff(sm) * fps
            vel_ext = np.maximum(vel, 0.0)
            if bottom < len(vel_ext):
                vel_ext[:bottom] = 0.0
            vel_sm = ScoringEngine._smooth(vel_ext, 3)
            vel_peaks = ScoringEngine._find_valid_peaks(
                vel_sm, min_prominence=30.0, min_distance=3)
            max_vel = float(np.max([vel_sm[p] for p in vel_peaks])) if vel_peaks else 0.0

            amp_score = 100.0 - abs(amplitude - ideal_amp) * 1.5
            amp_score = max(0.0, min(100.0, amp_score))
            vel_score = (max_vel / ideal_vel) * 100.0
            vel_score = max(0.0, min(100.0, vel_score))
            joint_score = amp_score * 0.5 + vel_score * 0.5
            total += joint_score

            html_rows += (
                f"<tr><td align='center'>{name}</td>"
                f"<td align='center'>{min_angle:.1f}°</td>"
                f"<td align='center'>{amplitude:.1f}°</td>"
                f"<td align='center'>{max_vel:.1f}°/s</td>"
                f"<td align='center'>{joint_score:.1f}</td></tr>"
            )

        total_score = total / len(joints)

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center'><b>关节</b></th>
                <th align='center'><b>最低屈曲角度</b></th>
                <th align='center'><b>屈伸幅度</b></th>
                <th align='center'><b>最大蹬伸角速度</b></th>
                <th align='center'><b>单项得分</b></th>
            </tr>
            {html_rows}
        </table>
        """
        return total_score, html_report

    # ============================================================
    # 动力链协同与发力节奏
    # ============================================================
    @staticmethod
    def compute_coordination(frame_metrics, fps=30.0, window=None):
        """动力链协同与发力节奏评估。

        发力时序基本固定（近端→远端：髋→膝→肩→肘→腕），因此：
          1. 只在下蹲最低点之后的蹬伸阶段寻找各关节伸展峰值；
          2. 额外计算各关节的启动时序（onset），与峰值时序共同反映动力链传导；
          3. 相邻环节时差在 2~4 帧同步窗口内视为同步，超出窗口或顺序颠倒则扣分。
        """
        if len(frame_metrics) < 10:
            return 0.0, "数据量过少，无法分析动力链"

        if window is None:
            window = ScoringEngine.CHAIN_WINDOW_FRAMES

        bottom = ScoringEngine._find_squat_bottom(frame_metrics)

        # 各环节：(名称, 序列, 峰值方向)。角度峰值=伸展到最大(max)，手腕峰值=出手释放(min)
        chain = [
            ("髋部伸展", ScoringEngine._aligned_angles(frame_metrics, 2), "max"),
            ("膝部蹬伸", ScoringEngine._aligned_angles(frame_metrics, 3), "max"),
            ("肩部发力", ScoringEngine._aligned_angles(frame_metrics, 0), "max"),
            ("肘部传递", ScoringEngine._aligned_angles(frame_metrics, 1), "max"),
            ("手腕释放", ScoringEngine._aligned_wrist_y(frame_metrics), "min"),
        ]

        onsets = []
        peaks = []
        for name, sig, mode in chain:
            onset, peak = ScoringEngine._extension_onset_peak(sig, bottom, window, mode)
            onsets.append(onset)
            peaks.append(peak)

        def chain_penalty(seq):
            """按理想顺序（髋→膝→肩→肘→腕）计算时序罚分。"""
            pen = 0.0
            for i in range(len(seq) - 1):
                lag = seq[i + 1] - seq[i]
                if lag < 0:
                    # 顺序颠倒（远端先于近端发力），重罚
                    pen += abs(lag) * 4.0
                elif lag > window:
                    # 传导过慢（超出 2~4 帧同步窗口），轻罚
                    pen += (lag - window) * 2.0
            return pen

        score = 100.0 - chain_penalty(peaks) - chain_penalty(onsets) * 0.5
        score = max(0.0, min(100.0, score))

        t_start = min(onsets)
        t_end = max(peaks)
        total_time = t_end - t_start if t_end > t_start else 1

        html_rows = ""
        for i, (name, sig, mode) in enumerate(chain):
            rel_pct = ((peaks[i] - t_start) / total_time) * 100
            html_rows += (
                f"<tr><td align='center'>{name}</td>"
                f"<td align='center'>第 {onsets[i]} 帧</td>"
                f"<td align='center'>第 {peaks[i]} 帧</td>"
                f"<td align='center'>{rel_pct:.1f}%</td></tr>"
            )

        report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center'><b>动力链环节</b></th>
                <th align='center'><b>启动时序</b></th>
                <th align='center'><b>伸展峰值</b></th>
                <th align='center'><b>峰值相对耗时</b></th>
            </tr>
            {html_rows}
        </table>
        <div align='center' style='color:#A6ADC8; font-size:12px; margin-top:6px;'>
            同步窗口 {window} 帧（2~4 帧）；理想传导顺序：髋 → 膝 → 肩 → 肘 → 腕
        </div>
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
        if not std_seqs:
            raise ScoringError(
                "标准序列库为空，无法选择冠军样本（输入数据异常）", kind="scoring")
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
        champ = ScoringEngine._validate_angle_sequence(champ, "冠军样本序列")
        test_seq = ScoringEngine._validate_angle_sequence(test_seq, "测试序列")
        # 特征维度不一致（如冠军 4 维、测试 5 维）→ 数组长度不一致
        if champ.shape[1] != test_seq.shape[1]:
            raise ScoringError(
                f"特征维度不一致：冠军 {champ.shape[1]} 维 vs 测试 {test_seq.shape[1]} 维"
                "（数组长度不一致）", kind="scoring")
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

    # ============================================================
    # 加权叠加
    # ============================================================
    @staticmethod
    def combine_scores(scores, weights):
        """按权重对多个 0~100 评分做加权叠加（权重自动归一化）。

        参数：
            scores  —— dict：模块名 -> 分数（0~100）
            weights —— dict：模块名 -> 权重（任意非负，按和归一化）

        返回：加权综合分（0~100）。只对两者共有且权重 > 0 的 key 求加权平均；
              若没有有效交集或权重和为 0，则退化为 scores 的算术平均。
        """
        if not scores:
            return 0.0
        common = [k for k in scores if k in weights and weights.get(k, 0) > 0]
        if not common:
            return float(np.mean(list(scores.values())))
        wsum = sum(weights[k] for k in common)
        if wsum <= 0:
            return float(np.mean([scores[k] for k in common]))
        return float(sum(weights[k] * scores[k] for k in common) / wsum)
