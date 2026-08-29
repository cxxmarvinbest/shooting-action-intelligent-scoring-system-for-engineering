# -*- coding: utf-8 -*-
"""
跨平台字体工具（QT_Linux/fonts）
==================================
职责：为 Qt 客户端选择当前平台可用的中文字体 / 等宽字体，避免硬编码 Windows 字体
      （Microsoft YaHei / Consolas）导致 Linux 上中文显示为方块。

优先级：
  - 中文：Noto Sans CJK SC → 思源黑体 → 文泉驿 → 微软雅黑 → 苹方（逐级兜底）
  - 等宽：DejaVu Sans Mono → Noto Sans Mono → Ubuntu Mono → Consolas → monospace

用法（须在 QApplication 创建之后调用，通常发生在 UI 构建 / 绘制阶段）：
    from fonts import cjk_font, mono_font
    label.setFont(cjk_font(12))

依赖：PyQt5
"""

from PyQt5.QtGui import QFont, QFontDatabase

# 中文无衬线候选（按优先级从高到低）
_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans CN", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Microsoft YaHei", "PingFang SC",
]

# 等宽字体候选（日志 / 代码块）
_MONO_CANDIDATES = [
    "DejaVu Sans Mono", "Noto Sans Mono", "Ubuntu Mono", "Liberation Mono",
    "Consolas", "Courier New", "monospace",
]

# 已安装字体族缓存（首次调用时扫描一次）
_installed = None


def _installed_families():
    """返回当前系统已安装字体族集合（缓存，避免每次 paint 重复扫描）。"""
    global _installed
    if _installed is None:
        _installed = set(QFontDatabase().families())
    return _installed


def _pick(candidates, fallback):
    """从候选字体族中选第一个已安装的，否则返回 fallback。"""
    installed = _installed_families()
    for name in candidates:
        if name in installed:
            return name
    return fallback


def cjk_font(point_size):
    """返回适配当前平台的中文字体（无衬线）。"""
    return QFont(_pick(_CJK_CANDIDATES, "Sans Serif"), point_size)


def mono_font(point_size):
    """返回适配当前平台的等宽字体。"""
    return QFont(_pick(_MONO_CANDIDATES, "monospace"), point_size)
