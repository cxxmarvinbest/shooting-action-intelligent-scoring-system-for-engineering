# -*- coding: utf-8 -*-
"""
全链路性能测速埋点（common/profiler）—— N2
=============================================
职责：对实时 RTSP 推理主链路逐帧统计各阶段耗时（ms）与模块吞吐（fps），
      每 N 帧周期性打印一次统计日志；不阻塞业务、埋点开销极小、可一键关闭。

设计要点（对齐 N2 验收）：
  1. 全局总开关：config/perf.yaml 的 PERF_ENABLED（实时）/ PERF_OFFLINE_ENABLED（离线）。
     - 关闭时 get_*_profiler() 返回 _NullProfiler 单例，其 begin/mark/zone/end 全部
       是 `pass` 的空方法（每帧仅几次空函数调用，约亚微秒级，相对 30ms 帧可忽略 ≈ 0 开销）；
     - 也可直接注释掉业务代码里的 `pf.*` 行彻底移除埋点（埋点均为独立单行，不嵌逻辑）。
  2. 逐帧六段耗时（ms）：
       preprocess  图像预处理（实时 lite=补黑边 copyMakeBorder；cpp 引擎无 Python 预处理≈0）
       detect      目标检测「RKNN 推理 + NMS 后处理 + anti-flicker 跟踪选框」
       pose        姿态估计「抠 ROI + RKNN 姿态推理 + 关键点后处理/坐标映射」
       feature     特征提取 extract_pose_features（关节角度/左右侧）
       fsm         ShotFSM 业务状态机 feed()（独立区间计时，不含帧缓存 copy 等杂项）
       total       单帧完整链路墙钟总耗时（含上述全部 + 帧缓存/AI 回写/编排杂项）
  3. 周期日志：每 PERF_REPORT_EVERY 帧打印最近 PERF_WINDOW 帧的 max/min/avg(ms) + 模块 fps
     （模块 fps = 1000/avg_ms，反映该段单独吞吐；total fps = 端到端实际帧率）。
  4. 链路隔离：实时链路用 get_realtime_profiler()，离线链路用 get_offline_profiler()，
     两者统计互不干扰；离线默认关闭（PERF_OFFLINE_ENABLED=false），按需复用。
  5. 不阻塞业务：仅 perf_counter 计时 + deque 累加，无锁、无 IO（打印走 logger，异常不抛出）。

计时 API（两种，均为单行埋点，不改动业务代码结构）：
  pf.begin_frame(idx)          # 帧开始（重置段计时基准）
  pf.mark("detect")            # 顺序打点：记录「上一个 mark/begin 到此刻」为该段耗时
  pf.zone_begin("fsm") / pf.zone_end("fsm")   # 显式区间（用于与 mark 链不连续的段，如 FSM）
  pf.end_frame()               # 帧结束：累计 total，到达周期则打印统计

说明：config 的 import 延迟到工厂函数内，使本模块可在无 config/无 RKNN 环境下被独立单测。
"""

import logging
import time
from collections import deque

logger = logging.getLogger("basketball_scoring")

# 统计的阶段顺序（total 单独追加在末尾展示）
STAGES = ("preprocess", "detect", "pose", "feature", "fsm")

# 阶段中文标签（日志展示用）
STAGE_LABELS = {
    "preprocess": "图像预处理",
    "detect": "检测(推理+后处理)",
    "pose": "姿态(RKNN+后处理)",
    "feature": "特征pose_feature",
    "fsm": "ShotFSM.feed",
    "total": "单帧总链路",
}


class FrameProfiler:
    """活跃埋点器：enabled=True。逐帧收集各段耗时，滑动窗口周期统计。"""

    enabled = True

    def __init__(self, name="实时RTSP", window=100, report_every=100):
        self.name = name
        self.window = max(10, int(window))
        self.report_every = max(1, int(report_every))
        # 每段一个有界 deque（滑动窗口，天然防内存无限增长）
        self._samples = {k: deque(maxlen=self.window) for k in STAGES + ("total",)}
        self._frame = 0
        self._t0 = 0.0      # 本帧起始 perf_counter
        self._last = 0.0    # 上一个顺序 mark 的 perf_counter
        self._zones = {}    # 显式区间：stage -> begin perf_counter

    # ---- 逐帧埋点 ----
    def begin_frame(self, idx=None):
        self._frame += 1
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._zones.clear()

    def mark(self, stage):
        """顺序打点：把「上一 mark/begin 到现在」的耗时(ms)计入 stage。"""
        now = time.perf_counter()
        self._samples[stage].append((now - self._last) * 1000.0)
        self._last = now

    def zone_begin(self, stage):
        """显式区间开始（与 mark 链独立，用于 FSM 等非连续段）。"""
        self._zones[stage] = time.perf_counter()

    def zone_end(self, stage):
        """显式区间结束：把区间耗时(ms)计入 stage。"""
        t = self._zones.pop(stage, None)
        if t is not None:
            self._samples[stage].append((time.perf_counter() - t) * 1000.0)

    def end_frame(self):
        """帧结束：计 total，到达周期则输出统计。"""
        self._samples["total"].append((time.perf_counter() - self._t0) * 1000.0)
        if self._frame % self.report_every == 0:
            self._report()

    def reset(self):
        for dq in self._samples.values():
            dq.clear()
        self._frame = 0

    # ---- 周期统计输出 ----
    def _report(self):
        try:
            lines = [
                "=" * 64,
                f"[性能埋点|{self.name}] 最近 {self.window} 帧滑动统计"
                f"（累计已处理 {self._frame} 帧，单位 ms / fps）",
                f"  {'阶段':<20}{'avg':>9}{'min':>9}{'max':>9}{'fps':>9}{'样本':>7}",
                "-" * 64,
            ]
            for k in STAGES + ("total",):
                dq = self._samples[k]
                n = len(dq)
                if n == 0:
                    continue
                avg = sum(dq) / n
                mn = min(dq)
                mx = max(dq)
                fps = (1000.0 / avg) if avg > 0 else 0.0
                prefix = ">>" if k == "total" else "  "
                lines.append(
                    f"{prefix}{STAGE_LABELS[k]:<18}{avg:>9.2f}{mn:>9.2f}"
                    f"{mx:>9.2f}{fps:>9.1f}{n:>7d}")
            lines.append("=" * 64)
            logger.info("\n".join(lines))
        except Exception as e:  # 埋点/统计自身异常绝不允许影响业务
            logger.debug("性能埋点统计输出失败: %s", e)


class _NullProfiler:
    """空埋点器：enabled=False。所有方法为空，关闭时每帧仅亚微秒级函数调用开销。"""

    enabled = False
    __slots__ = ()

    def begin_frame(self, *a, **k):
        pass

    def mark(self, *a, **k):
        pass

    def zone_begin(self, *a, **k):
        pass

    def zone_end(self, *a, **k):
        pass

    def end_frame(self, *a, **k):
        pass

    def reset(self, *a, **k):
        pass


# 模块级单例（Null 无状态共享一个；Active 每链路一个，统计互相隔离）
_NULL = _NullProfiler()
_rt_profiler = None
_off_profiler = None


def get_realtime_profiler():
    """实时 RTSP 链路埋点单例：PERF_ENABLED 为真返回 FrameProfiler，否则返回空埋点器。"""
    global _rt_profiler
    if _rt_profiler is None:
        from config import Config  # 延迟导入：保持本模块可独立单测
        if bool(Config.get("PERF_ENABLED", False)):
            _rt_profiler = FrameProfiler(
                name="实时RTSP",
                window=Config.get("PERF_WINDOW", 100),
                report_every=Config.get("PERF_REPORT_EVERY", 100))
            logger.info("性能埋点已启用（实时 RTSP 链路）：每 %s 帧打印一次，窗口 %s 帧",
                        Config.get("PERF_REPORT_EVERY", 100), Config.get("PERF_WINDOW", 100))
        else:
            _rt_profiler = _NULL
    return _rt_profiler


def get_offline_profiler():
    """离线视频分析链路埋点单例：PERF_OFFLINE_ENABLED 为真才启用（默认关闭）。"""
    global _off_profiler
    if _off_profiler is None:
        from config import Config
        if bool(Config.get("PERF_OFFLINE_ENABLED", False)):
            _off_profiler = FrameProfiler(
                name="离线视频",
                window=Config.get("PERF_WINDOW", 100),
                report_every=Config.get("PERF_REPORT_EVERY", 100))
            logger.info("性能埋点已启用（离线视频链路）：每 %s 帧打印一次",
                        Config.get("PERF_REPORT_EVERY", 100))
        else:
            _off_profiler = _NULL
    return _off_profiler
