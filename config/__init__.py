# -*- coding: utf-8 -*-
"""config 包：多 YAML 配置的统一访问入口。

用法：
    from config import Config
    print(Config.DET_RKNN_PATH)      # 属性式访问
    print(Config.SCORE_WEIGHTS)      # 嵌套 dict
"""

from .loader import Config, PROJECT_ROOT

__all__ = ["Config", "PROJECT_ROOT"]
