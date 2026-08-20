# -*- coding: utf-8 -*-
"""
RKNN 推理公共工具模块（common/rknn_utils）
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
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # Windows 本地调试时允许导入本模块但不实例化


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
def letterbox_to_model(frame, model_w, model_h, pad_color=(114, 114, 114)):
    """
    将任意尺寸帧等比缩放并补灰边到模型输入尺寸。
    返回 (输入图, scale, dw, dh)：scale 为缩放比，dw/dh 为左右/上下补边像素。
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

    def __init__(self, rknn_path, model_w=640, model_h=640):
        if RKNNLite is None:
            raise RuntimeError(
                "未安装 rknn-toolkit-lite2，本模块只能在 RK3588 端运行推理。\n"
                "安装: pip install rknn_toolkit_lite2-x.x.x-*.whl")
        # RKNNLite(lite2) 没有 get_model_attr 方法（该方法是 PC 端 rknn-toolkit2
        # 的 RKNN 类独有），输入尺寸须由调用方显式指定，默认 yolov8 的 640x640。
        self.model_w = model_w
        self.model_h = model_h
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 模型失败: {rknn_path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"初始化 RKNN 运行时失败: {rknn_path}")

    def infer(self, frame):
        """letterbox 到模型输入尺寸并推理，返回 (原始输出列表, scale, dw, dh)"""
        img, scale, dw, dh = letterbox_to_model(frame, self.model_w, self.model_h)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)  # rknn_lite 自动做 /255 归一化(依导出配置)
        outputs = self.rknn.inference(inputs=[img])
        return outputs, scale, dw, dh

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __del__(self):
        self.release()
