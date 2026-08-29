@echo off
REM ============================================================
REM  Qt 客户端打包脚本（在 QT 目录下双击或在命令行执行）
REM  前置：pip install pyinstaller pyqt6 requests
REM ============================================================

echo [1/2] 清理旧构建产物...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [2/2] 打包 exe（窗口程序，无控制台）...
pyinstaller --noconfirm --clean --windowed --name BasketballQtClient ^
    --collect-all PyQt6 ^
    main.py

echo.
echo 打包完成：dist\BasketballQtClient\BasketballQtClient.exe
pause
