# -*- coding: utf-8 -*-
"""
Qt 客户端入口（QT_Linux/main）
===============================
运行（RK3588 Linux 桌面环境）：
    cd QT_Linux
    ./run.sh
    或
    python3 main.py

算法服务地址（默认本机 127.0.0.1:8899），可通过环境变量覆盖：
    LQ_QT_HOST=192.168.8.75 LQ_QT_PORT=8899 python3 main.py

依赖：PyQt5 / requests（见 requirements.txt）
"""

import logging
import os
import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    host = os.environ.get("LQ_QT_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("LQ_QT_PORT", "8899"))
    except ValueError:
        port = 8899

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(host, port)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
