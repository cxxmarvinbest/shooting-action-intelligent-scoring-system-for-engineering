# -*- coding: utf-8 -*-
"""
轻量单目标跟踪层（detection/single_target_tracker）
======================================================
职责：对检测输出的「主球员框」做时序稳定，抑制 flicker，同时充当误检/漏检抑制器。

背景（为何需要）：
  现有 TargetSelector.select_main_player_box 是无状态、纯单帧的——每帧独立选
  「面积最大 player」。在以下场景下，主球员框会在目标间跳变 / 闪断，
  下游姿态估计随之产生关键点剧抖或整帧丢失：
    1. 身份跳变：多人同屏，两人面积帧间互换；
    2. 单帧误检：背景被误检成 player 且面积更大，抢走主球员身份 1 帧；
    3. 单帧漏检：置信度波动导致 player 漏检，姿态断链。

方案（纯 Python，零额外推理、零新依赖、无 NPU 开销）：
  IoU 匹配 + EMA 平滑 + hangover 兜底（一阶匀速外推）+ confirm 确认。

状态机：
  NO_TARGET ──首帧采信(面积最大候选)──> TENTATIVE
  TENTATIVE ──IoU 续上 × confirm 帧──> CONFIRMED
  TENTATIVE ──未续上──> 放弃（当前帧有候选则重新采信，否则回 NO_TARGET）
  CONFIRMED ──命中──> 保持（EMA 平滑 + 更新速度）
  CONFIRMED ──漏检(miss <= hangover)──> HOLDING（外推）
  CONFIRMED ──漏检(miss >  hangover)──> NO_TARGET（丢弃）
  HOLDING   ──命中──> CONFIRMED（恢复）
  HOLDING   ──漏检(miss >  hangover)──> NO_TARGET（丢弃）

坐标系约定：
  本类不关心具体坐标系，只要求「同一实例处理的所有帧处于同一坐标系」。
  离线链路（640x640 画布）与实时链路（640x360 det360）坐标不同，
  → 接入方必须为每条链路各实例化一个独立实例，不可跨链路共享。
  → 切换视频 / 启动实时流时调用 reset() 回到无目标状态。

依赖：无（纯 Python 标准库）。
"""


class SingleTargetTracker:
    """主球员单目标轻量跟踪器（IoU 匹配 + EMA + 外推兜底 + confirm）。

    用法：
        tracker = SingleTargetTracker(iou_thresh=0.3, hangover=3,
                                      confirm=1, ema_alpha=0.5)

        # 每帧（两种等价方式）：
        cands = SingleTargetTracker.extract_player_candidates(dets, player_cls_id=0)
        box, state = tracker.update(cands)
        # 或直接： box, state = tracker.update_from_dets(dets)

        # box: (x1,y1,x2,y2) 稳定主球员框 或 None
        # state: NO_TARGET / TENTATIVE / CONFIRMED / HOLDING
    """

    # 状态常量
    NO_TARGET = "NO_TARGET"    # 无目标（未采信，也未在兜底）
    TENTATIVE = "TENTATIVE"    # 暂定（首帧采信，尚未确认）
    CONFIRMED = "CONFIRMED"    # 锁定（已确认）
    HOLDING = "HOLDING"        # 兜底中（漏检，外推保持）

    def __init__(self, iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=0.5,
                 frame_size=None):
        """
        参数：
            iou_thresh —— 命中判定的 IoU 阈值（候选与锁定框）
            hangover   —— 连续漏检最多兜底的帧数（超过则丢弃重置）
            confirm    —— tentative 需连续命中多少帧才转 CONFIRMED
                          （confirm=1 表示「首帧采信 + 下一帧续上即锁定」）
            ema_alpha  —— 平滑系数（0~1，越大越跟手；1.0 = 不平滑）
            frame_size —— 可选 (w,h)，用于外推时 clamp 到画面内
        """
        self.iou_thresh = iou_thresh
        self.hangover = max(1, int(hangover))
        self.confirm = max(1, int(confirm))
        self.ema_alpha = ema_alpha
        self._frame_size = frame_size
        self.reset()

    # ==================== 对外接口 ====================

    def reset(self):
        """回到无目标状态。视频切换 / 实时流启动时必须调用。"""
        self.state = self.NO_TARGET
        self._locked_box = None        # 当前锁定框（已平滑）
        self._last_center = None       # 上一帧锁定框中心（算速度用）
        self._velocity = (0.0, 0.0)    # 帧间速度（外推用）
        self._miss_cnt = 0             # 连续未命中帧数
        self._confirm_cnt = 0          # tentative 阶段连续命中帧数

    def update_from_dets(self, dets, player_cls_id=0):
        """便捷入口：从检测结果直接更新（内部提取 player 候选）。"""
        return self.update(self.extract_player_candidates(dets, player_cls_id))

    def update(self, candidates, frame_size=None):
        """用当前帧 player 候选框更新跟踪状态。

        参数：
            candidates —— player 候选框列表 [(x1,y1,x2,y2), ...]（退化框会被剔除）
            frame_size —— 可选 (w,h)，覆盖构造时值，用于外推 clamp
        返回：
            (box, state)
              box   —— 稳定主球员框 (x1,y1,x2,y2)，无目标时为 None
              state —— NO_TARGET / TENTATIVE / CONFIRMED / HOLDING
        """
        if frame_size is not None:
            self._frame_size = frame_size
        cands = self._sanitize(candidates)

        if self.state == self.NO_TARGET:
            return self._on_no_target(cands)
        if self.state == self.TENTATIVE:
            return self._on_tentative(cands)
        # CONFIRMED / HOLDING 共用命中/兜底逻辑
        return self._on_tracking(cands)

    # ==================== 便捷属性 ====================

    @property
    def is_active(self):
        """是否有稳定目标可用（已锁定或兜底中）。"""
        return self.state in (self.CONFIRMED, self.HOLDING)

    @property
    def locked_box(self):
        return self._locked_box

    # ==================== 静态工具 ====================

    @staticmethod
    def extract_player_candidates(dets, player_cls_id=0):
        """从检测结果列表中提取 player 候选框（返回原始检测框，不做平滑）。"""
        return [tuple(d['box']) for d in dets if d['cls'] == player_cls_id]

    @staticmethod
    def _sanitize(candidates):
        """剔除退化框（宽/高 <= 0），统一转为 float。"""
        out = []
        for b in candidates:
            if len(b) != 4:
                continue
            x1, y1, x2, y2 = b
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            out.append((float(x1), float(y1), float(x2), float(y2)))
        return out

    @staticmethod
    def _area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aarea = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        barea = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = aarea + barea - inter
        return inter / (union + 1e-9)

    # ==================== 状态处理 ====================

    def _on_no_target(self, cands):
        """无目标：首帧采信面积最大候选，进入 tentative。"""
        if not cands:
            return None, self.NO_TARGET
        best = max(cands, key=self._area)
        self._locked_box = best
        self._last_center = self._center(best)
        self._velocity = (0.0, 0.0)
        self._miss_cnt = 0
        self._confirm_cnt = 0
        self.state = self.TENTATIVE
        return best, self.TENTATIVE

    def _on_tentative(self, cands):
        """暂定：续上即累加确认，续不上则放弃（防单帧误检）。"""
        best, _iou = self._match(cands, self._locked_box)
        if best is not None:
            self._miss_cnt = 0
            self._confirm_cnt += 1
            self._locked_box = self._smooth(self._locked_box, best)
            self._update_velocity(self._locked_box)
            if self._confirm_cnt >= self.confirm:
                self.state = self.CONFIRMED
            return self._locked_box, self.state
        # 未续上：放弃 tentative
        if cands:
            # 当前帧仍有候选 → 立即重新采信（回到 tentative 起点）
            return self._on_no_target(cands)
        self._reset_state()
        return None, self.NO_TARGET

    def _on_tracking(self, cands):
        """已锁定/兜底：命中则平滑，漏检则外推兜底，超限则丢弃。"""
        best, _iou = self._match(cands, self._locked_box)
        if best is not None:
            self._miss_cnt = 0
            self._confirm_cnt += 1
            self._locked_box = self._smooth(self._locked_box, best)
            self._update_velocity(self._locked_box)
            self.state = self.CONFIRMED
            return self._locked_box, self.CONFIRMED
        # 漏检：hangover 兜底（外推）
        self._miss_cnt += 1
        if self._miss_cnt <= self.hangover:
            self._locked_box = self._extrapolate()
            self.state = self.HOLDING
            return self._locked_box, self.HOLDING
        self._reset_state()
        return None, self.NO_TARGET

    # ==================== 内部工具 ====================

    def _match(self, cands, ref_box):
        """在候选中找与 ref_box IoU 最大的命中框（身份续接的核心）。

        返回 (best, iou)；无命中（都 < iou_thresh）返回 (None, 0.0)。
        """
        best, best_iou = None, 0.0
        for b in cands:
            iou = self._iou(ref_box, b)
            if iou > best_iou:
                best_iou = iou
                best = b
        if best is not None and best_iou >= self.iou_thresh:
            return best, best_iou
        return None, 0.0

    def _smooth(self, old_box, new_box):
        """对中心点 + 宽高做 EMA 平滑（alpha 控制响应速度）。"""
        if self.ema_alpha >= 1.0:
            return new_box
        if self.ema_alpha <= 0.0:
            return old_box
        ocx, ocy = self._center(old_box)
        ncx, ncy = self._center(new_box)
        ow, oh = old_box[2] - old_box[0], old_box[3] - old_box[1]
        nw, nh = new_box[2] - new_box[0], new_box[3] - new_box[1]
        a = self.ema_alpha
        cx = a * ncx + (1 - a) * ocx
        cy = a * ncy + (1 - a) * ocy
        w = a * nw + (1 - a) * ow
        h = a * nh + (1 - a) * oh
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _update_velocity(self, box):
        """用平滑后锁定框的中心更新帧间速度。"""
        center = self._center(box)
        if self._last_center is not None:
            self._velocity = (center[0] - self._last_center[0],
                              center[1] - self._last_center[1])
        self._last_center = center

    def _extrapolate(self):
        """一阶匀速外推：锁定框中心沿速度平移，尺寸不变，clamp 到画面内。"""
        x1, y1, x2, y2 = self._locked_box
        vx, vy = self._velocity
        cx = (x1 + x2) / 2.0 + vx
        cy = (y1 + y2) / 2.0 + vy
        w = x2 - x1
        h = y2 - y1
        if self._frame_size is not None:
            fw, fh = self._frame_size
            if fw > 0:
                cx = max(w / 2.0, min(fw - w / 2.0, cx))
            if fh > 0:
                cy = max(h / 2.0, min(fh - h / 2.0, cy))
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def _reset_state(self):
        self.state = self.NO_TARGET
        self._locked_box = None
        self._last_center = None
        self._velocity = (0.0, 0.0)
        self._miss_cnt = 0
        self._confirm_cnt = 0
