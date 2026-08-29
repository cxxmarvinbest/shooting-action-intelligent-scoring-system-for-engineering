# -*- coding: utf-8 -*-
"""
Qt 客户端入口（QT/main）
=========================
运行：
    cd QT
    python main.py

打包（自行用命令行，见 build_exe.bat / README.md）：
    pyinstaller --noconfirm --clean --windowed --name BasketballQtClient main.py

依赖：PyQt6 / requests（见 requirements.txt）
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
