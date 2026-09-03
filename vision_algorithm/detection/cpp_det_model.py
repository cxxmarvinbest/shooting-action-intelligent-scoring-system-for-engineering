# -*- coding: utf-8 -*-
"""
C++ 检测引擎封装（detection/cpp_det_model）
=============================================
职责：封装 pybind11 编译的 rknn_yolov8.YOLOv8Detector（C++ 后处理），
      替代 RKNNDetModel 的 Python numpy 后处理，消除后处理瓶颈
      （Python 后处理 ~8ms → C++ 后处理 ~0.4ms）。

与 RKNNDetModel（rknn_lite 引擎）的差异：
  1. 后处理（DFL 解码 / NMS / 反量化）在 C++ 完成，单帧后处理 ~0.4ms；
  2. 阈值硬编码在 C++ 内部（BOX_THRESH=0.25、NMS_THRESH=0.45），
     无法像 Python 那样对篮球单独放宽到 0.05（小目标召回需 A/B 重点验证）；
  3. 输入 uint8 BGR（is_rgb=False），内部自动 letterbox 补边到模型尺寸
     （BG_COLOR=114 灰边，注意与 Python 黑边 0 的差异，坐标反算不受影响）；
  4. 返回坐标为「输入图坐标系」（C++ 已反算），类别 id 直接来自模型 argmax
     （两类别自训练模型：0=player，1=basketball，非 COCO 80 类）。

对外暴露：CppDetModel
依赖：rknn_yolov8（pybind11 .so，仅 RK3588 端可 import；Windows 本地降级为 None）

坐标系约定：detect_360(img360) 返回的 box 坐标即 img360 输入坐标系，
            实时路径直接归一化（x/src_w, y/src_h）映射到预览，无需再减补边。
"""

import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger("basketball_scoring")

# 把本文件所在目录加入 sys.path：rknn_yolov8 的 pybind11 .so
# （rknn_yolov8.cpython-310-aarch64-linux-gnu.so）与本文件同目录部署，
# 否则 `from rknn_yolov8 import ...` 会因 .so 目录不在 sys.path 而 ImportError。
_SO_DIR = os.path.dirname(os.path.abspath(__file__))
if _SO_DIR not in sys.path:
    sys.path.insert(0, _SO_DIR)

# 降级导入：Windows 本地开发环境无 rknn_yolov8 .so，允许本模块 import 但不实例化。
try:
    from rknn_yolov8 import YOLOv8Detector
except ImportError:
    YOLOv8Detector = None


class CppDetModel:
    """YOLOv8 检测的 C++ 引擎封装（pybind11，RK3588 端专用）。

    用法：
        model = CppDetModel(rknn_path, core_mask=7)
        dets = model.detect_360(img360)   # img360: 640x360 BGR uint8
        # dets = [{'box': (x1,y1,x2,y2), 'cls': 0|1, 'conf': float}, ...]（降序）
    """

    def __init__(self, rknn_path, core_mask=0):
        self._detector = None   # 先置 None，避免 __del__/release 访问未定义属性
        if YOLOv8Detector is None:
            raise ImportError(
                "未找到 rknn_yolov8 模块（pybind11 .so），C++ 检测引擎仅能在 RK3588 端运行。\n"
                "已自动将本文件目录加入 sys.path。若仍失败，请确认：\n"
                "  1) rknn_yolov8.cpython-310-aarch64-linux-gnu.so 与 cpp_det_model.py 同目录；\n"
                "  2) 板端 python3 为 3.10 且 aarch64（.so 的 cpython-310-aarch64 tag 必须匹配）；\n"
                "  3) 用 python3（而非其它版本解释器）启动脚本。")
        self._detector = YOLOv8Detector()
        ret = self._detector.init(rknn_path, core_mask)
        if ret != 0:
            raise RuntimeError(
                f"C++ 检测引擎加载模型失败: {rknn_path} (ret={ret})")
        self.model_w = self._detector.get_model_width()
        self.model_h = self._detector.get_model_height()
        logger.info("C++ 检测引擎加载成功: %s (input=%dx%d, core_mask=%d)",
                    rknn_path, self.model_w, self.model_h, core_mask)

    def detect_360(self, img360):
        """对（RGA 等比缩放的）检测输入图做检测，C++ 内部自动 letterbox 补边。

        参数：
            img360 —— 任意尺寸 BGR uint8 numpy 图（实时路径传 640x360）。
                      C++ 侧 convert_image_with_letterbox 会等比缩放 + 补边到
                      model_w x model_h，并在后处理用 x_pad/y_pad/scale 反算。

        返回：list of dict {'box':(x1,y1,x2,y2), 'cls':int, 'conf':float}
              坐标为「输入图坐标系」，按置信度降序（首个即最优目标）。
        """
        if img360.dtype != np.uint8:
            img360 = img360.astype(np.uint8)
        if img360.ndim != 3:
            raise ValueError(f"检测输入必须是 3 维 HWC，实际 {img360.shape}")
        # pybind11 要求内存连续，避免 stride 不一致导致通道错位
        if not img360.flags['C_CONTIGUOUS']:
            img360 = np.ascontiguousarray(img360)

        start_time = int(time.time() * 1000)          # 毫秒（供 C++ 统计 convert 耗时）
        infer_result = self._detector.infer_array(img360, start_time, False)  # is_rgb=False(BGR)

        dets = []
        for r in infer_result.detections:
            dets.append({
                'box': (int(r.left), int(r.top), int(r.right), int(r.bottom)),
                'cls': int(r.cls_id),
                'conf': float(r.confidence),
            })
        return dets

    def detect_on_canvas(self, canvas):
        """兼容旧 rknn_lite 接口：对「已补边的 640x640 画布」做检测，返回画布坐标。

        离线视频分析路径 _extract_frame_metrics 依赖此方法。因画布本身就是模型
        输入尺寸（640x640），喂给 C++ 内部 letterbox 时 scale=1、x_pad=0、y_pad=0，
        坐标无损，故直接复用 detect_360（返回坐标即画布坐标系）。
        """
        return self.detect_360(canvas)

    def release(self):
        if self._detector is not None:
            try:
                self._detector.release()
            except Exception as e:
                logger.warning("C++ 检测引擎释放失败（%s: %s）", type(e).__name__, e)
            self._detector = None

    def __del__(self):
        self.release()
