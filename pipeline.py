# -*- coding: utf-8 -*-
"""
实时识别管道主入口（pipeline）
================================
主线程入口（长驻进程）：装配并协调各 controller 子线程，从「实时摄像头」
完成「边录制边分析 + 逐投篮评分 + HTTP 上报 APP」的完整闭环。

组件装配：
  StandardLibManage —— 同步预加载 RKNN 模型 + 标准视频库（启动期一次性）
  CameraManage     —— RTSP 拉流子线程（MPP 硬解优先）
  InferenceManage  —— 实时分析子线程（采样→切分→逐投评分→rotate）
  HttpManage       —— HTTP 服务子线程（与 APP 通信）

运行：
  python pipeline.py
退出：
  Ctrl+C 或 SIGTERM 触发优雅退出（停止分析→保存录制→关闭摄像头→关闭 HTTP）
"""

import logging
import signal
import threading

from config import Config
from common.logger import setup_logger
from controller.standard_lib_manage import StandardLibManage
from controller.camera_manage import CameraManage
from controller.inference_manage import InferenceManage
from controller.http_manage import HttpManage

logger = logging.getLogger("basketball_scoring")


def main():
    log_file = setup_logger(Config.LOG_DIR)
    logger.info("实时识别服务日志文件: %s", log_file)

    # 1. 预加载模型 + 标准库（同步阻塞，首次 /record 免等待）
    std_lib_mgr = StandardLibManage()
    std_lib_mgr.ensure_models()

    # 2. 装配子线程
    camera = CameraManage(std_lib_mgr.analyzer)
    inference = InferenceManage(
        std_lib_mgr.analyzer, std_lib_mgr.std_cache,
        std_lib_mgr.std_video_count, camera)
    http = HttpManage(std_lib_mgr, camera, inference)

    # 3. 启动 HTTP 服务子线程（对 APP 暴露控制接口）
    http.start()
    logger.info("实时识别服务已就绪（HTTP 端口 %d）", Config.HTTP_PORT)

    # 4. 主线程等待退出信号
    stop_event = threading.Event()

    def _sig_handler(signum, frame):
        logger.info("收到退出信号 %s，开始优雅退出...", signum)
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, AttributeError):
        pass  # 非主线程或平台不支持 SIGTERM 时忽略

    while not stop_event.wait(timeout=1):
        pass

    # 5. 优雅退出（顺序：停止分析 → 保存录制 → 关闭摄像头 → 关闭 HTTP）
    logger.info("正在停止实时分析线程...")
    inference.stop()
    logger.info("正在保存并关闭录制...")
    camera.recording.close()
    logger.info("正在关闭摄像头...")
    camera.close()
    logger.info("正在关闭 HTTP 服务...")
    http.stop()
    logger.info("正在释放 RKNN 模型（NPU 资源）...")
    std_lib_mgr.analyzer.release_models()
    logger.info("实时识别服务已退出")


if __name__ == "__main__":
    main()
