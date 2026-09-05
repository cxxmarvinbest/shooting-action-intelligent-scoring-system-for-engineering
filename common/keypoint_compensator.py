# -*- coding: utf-8 -*-
"""
N3 姿态关键点局部补偿算法（common/keypoint_compensator.py）
==========================================================

三帧窗口 [i-1, i, i+1] 线性插值，补偿姿态估计产生的单帧孤立跳变/消失关键点。
仅做局部单帧补偿，大幅度真实人体动作不触发补偿，避免篡改真实运动数据。

业务规则：
  1. 窗口：取 i-1、i、i+1 三帧同一关节点；
  2. 补偿条件（全部满足才触发）：
     a) i-1 与 i+1 帧该点坐标稳定（欧氏距离 < DIST_THRESHOLD）；
     b) 第 i 帧该点：conf 很低(<CONF_LOW)，或坐标发生巨大偏移(>JUMP_THRESHOLD)，
        或直接消失(坐标≈0)；
     c) i-1 与 i+1 帧该点置信度均为高置信(>=CONF_HIGH)；
  3. 满足条件 → 用 i-1 与 i+1 线性插值补齐第 i 帧；
  4. 不补偿条件（满足任一即保留原始点）：
     a) i-1 与 i+1 本身坐标变化幅度 >= DIST_THRESHOLD（大幅运动）；
     b) i-1 或 i+1 该点本身不可见/低置信；
  5. 边界：连续多帧丢失不做补偿（每点独立丢失计数器）；
  6. 重要：原始 kpts 不变，返回补偿后的副本。

应用时机：kpts/kpt_conf 提取完成后、extract_pose_features() 之前。
         原始数据仍写入 JSON / CSV，补偿副本仅供给 FSM 状态机。
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger("basketball_scoring")

# COCO 17 关键点数量
NUM_KEYPOINTS = 17


class KeypointCompensator:
    """三帧窗口关键点局部补偿器。

    维护一个固定长度为 3 的滑动窗口（最近 3 帧的 kpts + kpt_conf），
    对当前帧（窗口中间帧，索引 1）的每个关节点独立判断是否需要补偿。

    设计原则：
      - 关闭时（enabled=False）compensate() 直接返回原始 kpts 的浅拷贝，开销极小；
      - 开启时仅做 numpy 向量化运算 + 少量条件判断，单帧 < 0.1ms；
      - 不修改输入数组，始终返回新数组。
    """

    def __init__(self,
                 enabled=False,
                 dist_threshold=30.0,
                 jump_threshold=80.0,
                 conf_low=0.3,
                 conf_high=0.5,
                 log_detail=False):
        """
        参数：
            enabled         —— 总开关（yaml COMPENSATE_ENABLED）
            dist_threshold  —— 像素；i-1 与 i+1 同一点坐标欧氏距离阈值（< 此值视为"稳定"）
            jump_threshold  —— 像素；第 i 帧相对邻居的跳变距离阈值（> 此值视为"异常跳变"）
            conf_low        —— 低置信度阈值（< 此值视为"不可靠"，触发补偿候选）
            conf_high       —— 高置信度阈值（>= 此值视为"可靠"邻居）
            log_detail      —— 详细日志（打印每一帧每个点的补偿详情，调试用）
        """
        self.enabled = enabled
        self.dist_threshold = dist_threshold
        self.jump_threshold = jump_threshold
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.log_detail = log_detail

        # 三帧滑动窗口：每个元素为 (kpts, kpt_conf) 或 None（帧不足时）
        self._window = deque(maxlen=3)
        # 每个关节点的连续丢失计数器（用于边界条件：连续多帧丢失不补偿）
        self._miss_counts = np.zeros(NUM_KEYPOINTS, dtype=np.int32)
        # 帧计数器（用于日志）
        self._frame_idx = -1

    def reset(self):
        """重置窗口状态（切换视频/会话时调用）。"""
        self._window.clear()
        self._miss_counts[:] = 0
        self._frame_idx = -1

    def compensate(self, kpts, kpt_conf, frame_idx=None):
        """对单帧关键点执行补偿，返回补偿后的 kpts 副本。

        参数：
            kpts      —— (17,2) float32 ndarray，当前帧关键点坐标（可为 None）
            kpt_conf  —— (17,) float/list，当前帧关键点置信度（可为 None）
            frame_idx —— 帧号（可选，仅用于日志）

        返回：
            compensated_kpts —— (17,2) ndarray，补偿后的副本（未补偿点与原始一致）
        """
        self._frame_idx = frame_idx if frame_idx is not None else (self._frame_idx + 1)

        # ── 快速路径：关闭或无有效数据 ──
        if not self.enabled:
            return np.asarray(kpts, dtype=np.float32).copy() if kpts is not None else None

        if kpts is None:
            self._window.append((None, None))
            return None

        kpts_arr = np.asarray(kpts, dtype=np.float32)
        if kpts_arr.ndim != 2 or kpts_arr.shape[0] != NUM_KEYPOINTS:
            self._window.append((kpts_arr, kpt_conf))
            return kpts_arr.copy()

        # 处理 kpt_conf：统一为 (17,) float32
        if kpt_conf is not None:
            conf_arr = np.asarray(kpt_conf, dtype=np.float32).reshape(-1)
            if len(conf_arr) < NUM_KEYPOINTS:
                conf_arr = np.pad(conf_arr, (0, NUM_KEYPOINTS - len(conf_arr)), constant_values=0.0)
        else:
            conf_arr = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        # 把当前帧压入窗口
        self._window.append((kpts_arr.copy(), conf_arr.copy()))

        # 窗口不足 3 帧：无法做三帧补偿，直接返回副本
        if len(self._window) < 3:
            return kpts_arr.copy()

        # 取出三帧：[i-1, i, i+1]，当前帧是索引 1
        prev_kpts, prev_conf = self._window[0]
        curr_kpts, curr_conf = self._window[1]   # == (kpts_arr, conf_arr)
        next_kpts, next_conf = self._window[2]

        # 输出副本
        out = curr_kpts.copy()
        compensated_flags = np.zeros(NUM_KEYPOINTS, dtype=bool)

        # ── 逐关节点判断 ──
        for j in range(NUM_KEYPOINTS):
            result = self._compensate_single_joint(
                j, prev_kpts, prev_conf, curr_kpts, curr_conf, next_kpts, next_conf)
            if result is not None:
                out[j] = result
                compensated_flags[j] = True
                self._miss_counts[j] = 0   # 补偿成功，重置丢失计数
            else:
                # 判断当前帧该点是否为"丢失/异常"状态
                if self._is_abnormal(j, curr_kpts, curr_conf):
                    self._miss_counts[j] += 1
                else:
                    self._miss_counts[j] = 0

        # 日志输出
        if self.log_detail and compensated_flags.any():
            comp_joints = np.where(compensated_flags)[0].tolist()
            logger.debug(
                "N3补偿: frame=%d 补偿关节点=%s (共%d点)",
                self._frame_idx, comp_joints, len(comp_joints))
        elif any(compensated_flags):
            comp_joints = np.where(compensated_flags)[0].tolist()
            logger.info(
                "N3补偿: frame=%d 补偿关节点=%s (共%d点)",
                self._frame_idx, comp_joints, len(comp_joints))

        return out

    # ── 内部方法 ──

    def _compensate_single_joint(self, j, prev_kpts, prev_conf,
                                  curr_kpts, curr_conf, next_kpts, next_conf):
        """对单个关节点 j 执行三帧窗口补偿判断。返回插值坐标 (x,y) 或 None（不补偿）。"""

        # ── 条件 4b：邻居有效性检查 ──
        # i-1 或 i+1 本身不可见/低置信 → 不补偿
        if not self._is_reliable(j, prev_kpts, prev_conf):
            return None
        if not self._is_reliable(j, next_kpts, next_conf):
            return None

        # ── 条件 4a：邻居稳定性检查 ──
        # i-1 与 i+1 坐标距离 >= dist_threshold → 大幅运动，不补偿
        neighbor_dist = np.linalg.norm(prev_kpts[j] - next_kpts[j])
        if neighbor_dist >= self.dist_threshold:
            return None

        # ── 条件 2b：当前帧是否异常 ──
        # 当前帧该点必须异常（低置信/大跳变/消失）才考虑补偿
        if not self._is_abnormal(j, curr_kpts, curr_conf):
            return None  # 当前帧正常，无需补偿

        # ── 边界条件：连续多帧丢失不补偿 ──
        if self._miss_counts[j] >= 2:
            return None  # 已连续 3 帧以上异常（含本帧），视为持续丢失

        # ── 全部通过 → 线性插值 ──
        interpolated = (prev_kpts[j] + next_kpts[j]) / 2.0
        return interpolated

    def _is_reliable(self, j, kpts, conf):
        """判断第 j 个关节点是否为可靠的高置信点（邻居条件）。"""
        if conf is None or j >= len(conf):
            return False
        if conf[j] < self.conf_high:
            return False
        # 坐标不能是 (0,0) 这种"消失"态
        if kpts is None or j >= len(kpts):
            return False
        if kpts[j, 0] <= 0 and kpts[j, 1] <= 0:
            return False
        return True

    def _is_abnormal(self, j, kpts, conf):
        """判断第 j 个关节点在当前帧是否异常（低置信/大跳变/消失）。"""
        # 消失：(0,0) 或接近零
        if kpts is not None and j < len(kpts):
            if kpts[j, 0] <= 1.0 and kpts[j, 1] <= 1.0:
                return True

        # 低置信
        if conf is not None and j < len(conf):
            if conf[j] < self.conf_low:
                return True

        # 大跳变：需要与邻居比较（调用方已保证邻居可靠，这里检查相对位移）
        if (kpts is not None and len(self._window) >= 3
                and self._window[0][0] is not None and self._window[2][0] is not None):
            prev_kpts = self._window[0][0]
            next_kpts = self._window[2][0]
            # 当前点到邻居中点的距离
            mid = (prev_kpts[j] + next_kpts[j]) / 2.0
            jump_dist = np.linalg.norm(kpts[j] - mid)
            if jump_dist > self.jump_threshold:
                return True

        return False


class _NullCompensator:
    """空补偿器（COMPENSATE_ENABLED=false 时使用，所有方法为最小开销透传）。"""

    def __init__(self):
        self.enabled = False

    def reset(self):
        pass

    def compensate(self, kpts, kpt_conf=None, frame_idx=None):
        if kpts is None:
            return None
        return np.asarray(kpts, dtype=np.float32).copy()


def create_compensator_from_config():
    """从 Config 创建补偿器实例（工厂函数，全局单例模式由调用方控制）。

    返回：
        KeypointCompensator（开启时）或 _NullCompensator（关闭时）
    """
    try:
        from config import Config
        enabled = bool(Config.get("COMPENSATE_ENABLED", False))
        if not enabled:
            return _NullCompensator()
        return KeypointCompensator(
            enabled=True,
            dist_threshold=float(Config.get("COMPENSATE_DIST_THRESHOLD", 30.0)),
            jump_threshold=float(Config.get("COMPENSATE_JUMP_THRESHOLD", 80.0)),
            conf_low=float(Config.get("COMPENSATE_CONF_LOW", 0.3)),
            conf_high=float(Config.get("COMPENSATE_CONF_HIGH", 0.5)),
            log_detail=bool(Config.get("COMPENSATE_LOG_DETAIL", False)),
        )
    except Exception:
        # 配置加载失败时返回空补偿器，不阻塞主流程
        return _NullCompensator()
