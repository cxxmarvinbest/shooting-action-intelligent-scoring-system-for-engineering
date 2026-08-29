# -*- coding: utf-8 -*-
"""
子线程基类（common/thread_base）
==================================
所有 controller/*_manage.py 继承本基类，统一子线程生命周期管理：
  - start()：幂等启动（重复调用不重复起线程）
  - stop() ：请求退出（置位 stop_event）
  - join() ：等待线程结束
  - on_error()：线程异常回调（子类可选实现）
  - _run() ：子类实现的主循环，用 self.is_stopped() 判断是否退出

好处：所有控制子线程的「启动/停止/异常兜底」逻辑一致，新增子线程只需
继承并实现 _run()，主入口 pipeline.py 里 .start()/.stop() 即可。
"""

import logging
import threading

logger = logging.getLogger("basketball_scoring")


class ThreadBase:
    """子线程基类（统一生命周期 + 异常兜底）。"""

    def __init__(self, name=None):
        self._name = name or self.__class__.__name__
        self._thread = None
        self._stop_event = threading.Event()

    @property
    def name(self):
        return self._name

    # ------------------------------------------------------------------
    # 对外控制
    # ------------------------------------------------------------------
    def start(self, join_timeout=None):
        """启动子线程（幂等；stop 后可重新 start）。

        join_timeout：等待旧线程退出时的超时（秒）。None 表示一直等到退出；
        传入超时值可避免旧线程退出过慢（如 MPP 解码器清理需要数秒）时把调用方卡死。
        """
        if self._thread is not None and self._thread.is_alive():
            if self._stop_event.is_set():
                # 已请求停止但线程仍在退出中：等它退出后再重启（可限时）
                self._thread.join(join_timeout)
            else:
                return  # 正常运行中，幂等返回
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._safe_run, name=self._name, daemon=True)
        self._thread.start()
        logger.info("[%s] 子线程已启动", self._name)

    def stop(self):
        """请求停止子线程。"""
        self._stop_event.set()

    def is_stopped(self):
        """是否已收到停止请求（子类主循环用此判断退出）。"""
        return self._stop_event.is_set()

    def join(self, timeout=None):
        """等待子线程结束。"""
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _safe_run(self):
        try:
            self._run()
        except Exception as e:
            logger.error("[%s] 子线程异常退出: %s", self._name, e, exc_info=True)
            self.on_error(e)
        finally:
            logger.info("[%s] 子线程已退出", self._name)

    def _run(self):
        """子类实现：线程主循环。"""
        raise NotImplementedError("子类必须实现 _run()")

    def on_error(self, exc):
        """子类可选实现：线程异常回调（默认忽略）。"""
        pass
