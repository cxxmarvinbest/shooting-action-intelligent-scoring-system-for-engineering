# -*- coding: utf-8 -*-
"""
RKNN 推理公共工具模块（common/rknn_infer）
============================================
职责：提供 YOLOv8 检测与姿态模型共用的底层能力：
  1. 数值工具：sigmoid / softmax
  2. YOLOv8 后处理：DFL 解码 / 框解码(box_process) / 关键点解码(kps_process) / NMS
  3. 模型输入 letterbox 与输出张量 NCHW 规范化
  4. RKNN 基类 _RKNNBase：模型加载 / 推理 / 输入尺寸查询

被 tracker.det_model 与 pose_estimate.pose_model 共同依赖，不反向依赖任何业务模块。

后处理遵循 rknn_model_zoo 官方示例约定：
  - 检测：每尺度 (box_dfl, cls) 两路输出，共 6 路（DFL reg_max=16）
  - 姿态：每尺度 (box_dfl, cls, kps) 三路输出，共 9 路（kps=17*3=51 通道）
  - rknn_lite.inference 返回 NHWC，本模块自动转 NCHW

依赖：numpy / cv2 / rknn-toolkit-lite2（仅 RK3588 端可装）
"""

import cv2
import logging
import numpy as np

from common.exceptions import RknnInferenceError

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # Windows 本地调试时允许导入本模块但不实例化

logger = logging.getLogger("basketball_scoring")


# NPU 核心位掩码（与 RKNNLite.NPU_CORE_* / 示例 yolov8_py.py 保持一致）
_NPU_CORE_MASKS = {
    "AUTO": 0,      # 自动调度
    "0": 1,         # 仅 Core0
    "1": 2,         # 仅 Core1
    "2": 4,         # 仅 Core2
    "0_1": 3,       # Core0 + Core1
    "0_1_2": 7,     # Core0 + Core1 + Core2（三核全开）
}


def _resolve_core_mask(core_mask):
    """把 YAML/环境变量里的值解析为 NPU 核心位掩码整数。

    支持：
      - 整数位掩码：0 / 1 / 2 / 4 / 3 / 7（与示例代码 coreMask 一致）
      - 字符串："auto" / "0" / "1" / "2" / "0_1" / "0_1_2"
      - None：默认三核全开 7
    返回整数；若 RKNNLite 不可用则返回 None（仅 Windows 本地调试场景）。
    """
    if RKNNLite is None:
        return None

    # 已经是整数：直接透传（示例代码风格）
    if isinstance(core_mask, int):
        if 0 <= core_mask <= 7:
            return core_mask
        logger.warning(
            "NPU_CORE_MASK 整数 %d 超出有效范围 0~7，回退到三核全开 7", core_mask)
        return 7

    # 字符串/None：按名字映射
    key = "0_1_2" if core_mask is None else str(core_mask).strip().upper()
    if key not in _NPU_CORE_MASKS:
        logger.warning(
            "未知的 NPU_CORE_MASK 值 '%s'，回退到三核全开 7", core_mask)
        key = "0_1_2"
    return _NPU_CORE_MASKS[key]


# ============================================================
# 数值工具
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================
# YOLOv8 后处理
# ============================================================
def dfl(position):
    """Distribution Focal Loss 解码：(1, 4*reg_max, h, w) -> (1, 4, h, w) ltrb 距离"""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num  # reg_max = 16
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def make_grid_and_stride(grid_h, grid_w, img_h, img_w):
    """生成特征图网格坐标与对应步长"""
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_w / grid_w, img_h / grid_h],
                      dtype=np.float32).reshape(1, 2, 1, 1)
    return grid, stride


def box_process(position, img_h, img_w):
    """DFL 框解码：ltrb 距离 -> 输入图坐标系下的 xyxy"""
    grid_h, grid_w = position.shape[2:4]
    grid, stride = make_grid_and_stride(grid_h, grid_w, img_h, img_w)
    position = dfl(position)
    box_xy1 = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)


def kps_process(position, img_h, img_w):
    """姿态关键点解码：(1, 51, h, w) -> (1, 17, 3, h, w)，x/y 为输入图坐标，第 3 维为置信度"""
    grid_h, grid_w = position.shape[2:4]
    grid, stride = make_grid_and_stride(grid_h, grid_w, img_h, img_w)
    position = position.reshape(1, 17, 3, grid_h, grid_w).astype(np.float32)
    # 与 ultralytics 导出一致的解码公式: x = (raw * 2 + grid - 0.5) * stride
    position[:, :, 0, :, :] = (position[:, :, 0, :, :] * 2.0 + (grid[:, 0:1] - 0.5)) * stride[:, 0:1]
    position[:, :, 1, :, :] = (position[:, :, 1, :, :] * 2.0 + (grid[:, 1:2] - 0.5)) * stride[:, 1:2]
    position[:, :, 2, :, :] = sigmoid(position[:, :, 2, :, :])
    return position


def nms(boxes, scores, iou_thres):
    """标准 NMS。boxes: (N,4) xyxy, scores: (N,)。返回保留的索引列表"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


# ============================================================
# 模型输入/输出处理
# ============================================================
def letterbox_to_model(frame, model_w, model_h, pad_color=(0, 0, 0)):
    """
    将任意尺寸帧「长边缩到 model、短边等比缩放、多余补边」到模型输入尺寸。
    返回 (画布BGR图, scale, dw, dh)：scale 为缩放比，dw/dh 为左右/上下补边像素。
    默认补黑边（描黑边），省去 YOLO 内部再对原图做 letterbox 的耗时。
    """
    h, w = frame.shape[:2]
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((model_h, model_w, 3), pad_color, dtype=np.uint8)
    dw = (model_w - new_w) // 2
    dh = (model_h - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, scale, dw, dh


# rknn_lite 推理输入布局：模型虽源自 NCHW 的 ONNX，但编译成 rknn 后 API 输入为 NHWC
# （runtime 日志「framework layout: NCHW」是 ONNX 源布局，实际 `rknn.inference` 要 NHWC）。
# 因此这里用 NHWC 布局，且不显式传 data_format（rknn_lite 默认即 nhwc）。
INPUT_DATA_FORMAT = "nhwc"


def canvas_to_input(canvas):
    """把 letterbox 后的 BGR 画布转为模型输入 NHWC (1, H, W, 3) uint8-RGB。

    uint8 不除以 255（模型量化时已内置 mean=[0,0,0]/std=[255,255,255] 归一化）。
    """
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8 RGB
    return np.expand_dims(img, 0)                  # (1, H, W, 3) NHWC


def preprocess_to_input(frame, model_w, model_h, pad_color=(0, 0, 0)):
    """把单帧预处理为模型输入 NHWC (1, H, W, 3)（letterbox + BGR2RGB）。

    含推理输入校验（frame 为空 / 维度非法 / dtype 非法）。返回 (img, scale, dw, dh)。
    供检测/姿态两个模型共享同一份 letterbox 输入，避免重复 resize/cvtColor。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        raise RknnInferenceError("推理输入异常：frame 为空", kind="rknn")
    if getattr(frame, "ndim", 0) < 2:
        raise RknnInferenceError("推理输入异常：frame 维度非法", kind="rknn")
    if frame.dtype != np.uint8:
        raise RknnInferenceError(
            f"推理输入异常：frame dtype 应为 uint8，实际 {frame.dtype}", kind="rknn")

    canvas, scale, dw, dh = letterbox_to_model(frame, model_w, model_h, pad_color)
    img = canvas_to_input(canvas)
    return img, scale, dw, dh


def ensure_nchw(out, channel_size):
    """
    将单个输出张量规范为 NCHW。
    rknn_lite 返回 NHWC，通道轴按 channel_size 定位（box=64, kps=51, cls=类别数）。
    """
    if out.ndim != 4:
        return out
    if out.shape[1] == channel_size:
        return out                      # 已是 NCHW
    if out.shape[3] == channel_size:
        return np.transpose(out, (0, 3, 1, 2))
    # 兜底：取最小的非 batch 轴当通道轴
    axis = int(np.argmin(out.shape[1:])) + 1
    if axis != 1:
        out = np.moveaxis(out, axis, 1)
    return out


def as_nchw(out, is_nhwc=None):
    """
    按「统一布局」把单个 4D 输出张量转为 NCHW。

    rknn_lite 输出的所有张量布局一致：要么都是 NHWC（通道在最后一维），
    要么都是 NCHW。用 box 分支（64/65 通道）判定一次 is_nhwc 后，
    对每个输出统一处理即可——这样能正确处理「通道数大于空间边长」的小尺度
    （如 20×20 且通道 64/65），避免 argmin 误判。

    参数：
        is_nhwc —— True 表示通道在最后一维（转置）；False/None 视为 NCHW 原样返回。
    """
    if out.ndim != 4:
        return out
    if is_nhwc:
        return np.transpose(out, (0, 3, 1, 2))
    return out


# ============================================================
# RKNN 基类
# ============================================================
class _RKNNBase:
    """负责模型加载/推理/输入尺寸查询，后处理由子类实现"""

    def __init__(self, rknn_path, model_w=640, model_h=640, core_mask=None):
        if RKNNLite is None:
            raise RknnInferenceError(
                "未安装 rknn-toolkit-lite2，本模块只能在 RK3588 端运行推理。\n"
                "安装: pip install rknn_toolkit_lite2-x.x.x-*.whl",
                kind="rknn")
        # RKNNLite(lite2) 没有 get_model_attr 方法（该方法是 PC 端 rknn-toolkit2
        # 的 RKNN 类独有），输入尺寸须由调用方显式指定，默认 yolov8 的 640x640。
        self.model_w = model_w
        self.model_h = model_h
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RknnInferenceError(
                f"加载 RKNN 模型失败: {rknn_path}", kind="rknn", errno=ret)

        # RK3588 有三个 NPU 核心，默认 init_runtime 通常只使用 core 0。
        # 通过 core_mask 指定多核并行，可显著提升检测+姿态的推理吞吐。
        resolved_mask = _resolve_core_mask(core_mask)
        try:
            if resolved_mask is not None:
                ret = self.rknn.init_runtime(core_mask=resolved_mask)
                logger.info(
                    "RKNN 初始化使用核心掩码: 0x%02x (core_mask=%s)",
                    resolved_mask, core_mask if core_mask is not None else "0_1_2")
            else:
                ret = self.rknn.init_runtime()
        except TypeError as e:
            # 旧版 rknn-toolkit-lite2 可能不接受 core_mask 关键字
            logger.warning(
                "当前 RKNNLite.init_runtime 不支持 core_mask（%s），回退到默认单核模式", e)
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RknnInferenceError(
                f"初始化 RKNN 运行时失败（NPU 资源异常）: {rknn_path}",
                kind="rknn", errno=ret)

    def infer(self, frame):
        """letterbox 到模型输入尺寸并推理，返回 (原始输出列表, scale, dw, dh)。

        异常捕获：
          - 推理输入异常：frame 为空 / 维度非法 / dtype 非法
          - NPU 资源异常：self.rknn.inference 失败
        """
        img, scale, dw, dh = preprocess_to_input(frame, self.model_w, self.model_h)
        outputs = self.infer_input(img)
        return outputs, scale, dw, dh

    def infer_input(self, img):
        """对已预处理(letterbox+BGR2RGB+NCHW)的输入直接推理，返回原始输出列表。

        供调用方在「检测+姿态两个模型共享同一 letterbox 输入」时复用，
        避免对同一帧重复做 resize/cvtColor（减少不必要的重复计算）。
        """
        try:
            outputs = self.rknn.inference(inputs=[img], data_format=INPUT_DATA_FORMAT)
        except Exception as e:
            raise RknnInferenceError(
                "RKNN 推理失败（NPU 资源异常）", kind="rknn", cause=e) from e
        if outputs is None:
            raise RknnInferenceError(
                "RKNN 推理返回空输出（NPU 资源异常）", kind="rknn")
        return outputs

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __del__(self):
        self.release()
