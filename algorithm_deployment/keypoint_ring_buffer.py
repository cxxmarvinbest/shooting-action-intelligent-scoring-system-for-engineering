# -*- coding: utf-8 -*-
"""
关键点环形缓存 + 投篮动作切分模块（keypoint_ring_buffer）
=========================================================
职责：
  1. KeypointRingBuffer —— 固定容量环形队列（deque，滚动覆盖，永不关闭），
     只缓存每帧的「特征 dict」（不缓存原始图像），从根源上防止内存耗尽。
  2. ShotSegmenter —— 基于环形缓存的投篮事件状态机：
        IDLE → HOLD_PENDING → HOLDING →（手腕举到最高点=出手）→ 回退取窗口反向回溯起点
     在持续滚动的缓存里检测多次投篮，逐投切分出完整动作段。

设计要点（对应需求）：
  - 预缓存 35 帧：确认持球前要求环形缓存已积累至少 RING_PRECACHE_FRAMES 帧历史。
  - 环形队列永远滚动覆盖，deque(maxlen) 有界，天然不堆积、不耗尽内存。
  - 持球防抖：连续 HOLD_DEBOUNCE_FRAMES 帧判为持球才确认。
  - 持球超时：HOLD_TIMEOUT_FRAMES 帧内未触发出手即放弃，不挂死在未闭合动作。
  - 出手事件：wrist_y 在滑动窗口内出现极小值拐点（手腕举到最高点后回升）。
  - 滑动窗口回退 + 反向回溯：出手后回退 LOOKBACK_WINDOW_FRAMES 帧，
    反向找「球与人框首次相交」的真实动作起点。

对外暴露：KeypointRingBuffer / ShotSegmenter
依赖：config
"""

from collections import deque

from config import Config


class KeypointRingBuffer:
    """关键点环形缓存：固定容量、滚动覆盖、只存特征 dict（不含原始图像）。"""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen or Config.RING_MAX_FRAMES
        self._buf = deque(maxlen=self.maxlen)

    def push(self, frame_data):
        """写入一帧特征 dict；超过容量自动覆盖最旧帧。"""
        self._buf.append(frame_data)

    def snapshot(self):
        """返回当前缓存全部帧（按时间升序）的 list。"""
        return list(self._buf)

    def backtrack(self, n):
        """回退取最近 n 帧（按时间升序）；n 超过已有帧数时返回全部。"""
        return list(self._buf)[-n:]

    def clear(self):
        """清空缓存（录制/视频结束调用）。"""
        self._buf.clear()

    def __len__(self):
        return len(self._buf)


class ShotSegmenter:
    """投篮动作切分器：持球防抖 + 超时兜底 + 手腕最高点出手检测 + 滑动窗口反向回溯。"""

    IDLE = 0          # 未持球，等待持球
    HOLD_PENDING = 1  # 疑似持球（防抖计数中）
    HOLDING = 2       # 已确认持球（计时 + 追踪出手）

    def __init__(self, ring=None,
                 precache=None,
                 debounce=None,
                 timeout=None,
                 lookback=None,
                 release_window=None,
                 iou_margin=None):
        self.ring = ring or KeypointRingBuffer()
        self.precache = precache if precache is not None else Config.RING_PRECACHE_FRAMES
        self.debounce = debounce if debounce is not None else Config.HOLD_DEBOUNCE_FRAMES
        self.timeout = timeout if timeout is not None else Config.HOLD_TIMEOUT_FRAMES
        self.lookback = lookback if lookback is not None else Config.LOOKBACK_WINDOW_FRAMES
        self.release_window = (release_window if release_window is not None
                               else Config.RELEASE_TRIGGER_WINDOW)
        self.iou_margin = iou_margin if iou_margin is not None else Config.HOLD_IOU_MARGIN

        self.state = self.IDLE
        self.hold_count = 0   # 连续持球帧计数（防抖）
        self.hold_frames = 0  # 确认持球后的帧计数（超时）
        self.wbuf = []        # (idx, wrist_y) 出手检测滑动窗口
        self.shot_count = 0   # 已切出的投篮数量（编号）

    # ------------------------------------------------------------------
    # 持球判定：球框与球员框相交（含少量外扩余量吸收检测抖动）
    # ------------------------------------------------------------------
    def _is_ball_held(self, fd):
        pbox = fd.get('player_box')
        balls = fd.get('ball_boxes') or []
        if pbox is None or not balls:
            return False
        px1, py1, px2, py2 = pbox
        m = self.iou_margin
        px1, py1, px2, py2 = px1 - m, py1 - m, px2 + m, py2 + m
        for bx in balls:
            bx1, by1, bx2, by2 = bx
            # 1) 两框存在相交面积
            ix1, iy1 = max(px1, bx1), max(py1, by1)
            ix2, iy2 = min(px2, bx2), min(py2, by2)
            if ix1 < ix2 and iy1 < iy2:
                return True
            # 2) 球中心落在球员（外扩）框内
            bxc, byc = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if px1 <= bxc <= px2 and py1 <= byc <= py2:
                return True
        return False

    def _push_wrist(self, fd):
        self.wbuf.append((fd['idx'], fd.get('wrist_y')))
        if len(self.wbuf) > self.release_window:
            self.wbuf.pop(0)

    # ------------------------------------------------------------------
    # 出手检测：手腕举到最高点（wrist_y 极小值拐点）
    # ------------------------------------------------------------------
    def _detect_release(self):
        if len(self.wbuf) < self.release_window:
            return None
        ys = [w for _, w in self.wbuf]
        if any(y is None for y in ys):
            return None  # 关键点缺失，无法判定（交由超时兜底）
        min_i = ys.index(min(ys))
        # 极小值不能是窗口最左端（否则仍在单调上升中，尚未到拐点）
        if min_i == 0:
            return None
        # 极小值之后需至少 2 帧回升，确认手腕已越过最高点开始下落
        if min_i <= len(ys) - 3 and ys[min_i + 1] > ys[min_i] and ys[min_i + 2] > ys[min_i]:
            return self.wbuf[min_i][0]
        return None

    # ------------------------------------------------------------------
    # 滑动窗口回退 + 反向回溯真实动作起点
    # ------------------------------------------------------------------
    def _find_action_start(self, window):
        """在窗口内反向回溯：找最后一个「不持球」帧，其后的帧即真实动作起点（持球开始）。"""
        last_not_held = None
        for m in window:
            if not self._is_ball_held(m):
                last_not_held = m['idx']
        if not window:
            return 0
        if last_not_held is None:
            # 整个窗口都处于持球，起点取窗口首帧
            return window[0]['idx']
        for m in window:
            if m['idx'] > last_not_held:
                return m['idx']
        return window[-1]['idx']

    def _build_segment(self, release_idx):
        # 回退取最大窗口：至少保证预缓存 precache 帧历史可回溯
        window = self.ring.backtrack(max(self.lookback, self.precache))
        # 只考虑出手帧及之前的帧：出手后球若仍被误判为持球，反向回溯会把
        # 起点错误地定位到出手之后（start > release），必须排除。
        window_before = [m for m in window if m['idx'] <= release_idx]
        if not window_before:
            window_before = window
        start_idx = self._find_action_start(window_before)
        # 兜底：起点绝不晚于出手帧
        start_idx = min(start_idx, release_idx)
        metrics = [m for m in window if start_idx <= m['idx'] <= release_idx]
        if not metrics:
            metrics = window_before
        self.shot_count += 1
        return {
            'shot_idx': self.shot_count,
            'start_idx': start_idx,
            'release_idx': release_idx,
            'frame_metrics': metrics,
        }

    def _reset(self):
        self.state = self.IDLE
        self.hold_count = 0
        self.hold_frames = 0
        self.wbuf = []

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def feed(self, fd):
        """喂入一帧特征 dict；若检测到一次完整投篮则返回 ShotSegment dict，否则 None。"""
        self.ring.push(fd)
        held = self._is_ball_held(fd)

        if self.state == self.IDLE:
            if held:
                self.state = self.HOLD_PENDING
                self.hold_count = 1
            return None

        if self.state == self.HOLD_PENDING:
            if held:
                self.hold_count += 1
                if self.hold_count >= self.debounce:
                    # 连续 N 帧判为持球 -> 确认进入持球状态，开始计时 + 追踪出手
                    self.state = self.HOLDING
                    self.hold_frames = 0
                    self.wbuf = []
                    self._push_wrist(fd)
            else:
                self._reset()  # 防抖中断，回到未持球
            return None

        # HOLDING
        self.hold_frames += 1
        self._push_wrist(fd)

        if self.hold_frames > self.timeout:
            self._reset()  # 超时未出手，放弃该候选（不挂死在未闭合动作）
            return None

        release_idx = self._detect_release()
        if release_idx is not None:
            seg = self._build_segment(release_idx)
            self._reset()
            return seg
        return None

    def finalize(self):
        """录制/视频结束：清空缓存与状态。"""
        self._reset()
        self.ring.clear()
