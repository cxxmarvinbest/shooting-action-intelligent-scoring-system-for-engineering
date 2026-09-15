# -*- coding: utf-8 -*-
"""
pose 4 输出解析工具（test 专用，复现业务 4 输出后处理，不改业务主代码）
========================================================================
业务姿态模型（yolov8n-pose, flatten=True）导出 **4 个输出**：
  outputs[0..2] : 3 个尺度各一路 box+cls（box 64 通道 + cls 1 通道 = 65 合并，
                  或 box/cls 分离）。三个尺度 stride 8/16/32（大尺度在前）。
  outputs[3]    : 1 路 flatten 独立关键点输出（17 点 × (x,y,conf)）。

关键点必须从 outputs[3] 解析。旧「3 输出 / 9 输出（每尺度 box+cls+kps 三路）」
后处理会把关键点取空，导致 17 点全 0 —— 这是必须用第 4 路输出的原因。

坐标约定：
  - 关键点 x/y 为 **pose 输入图（model_w × model_h）坐标系**（已解码），
    调用方需反 letterbox（(x - dw)/scale, (y - dh)/scale）映射回 crop 图坐标，
    再加 crop 偏移回原图 —— 与业务 video_analyzer 的坐标链路一致。
  - box 输出同样为 pose 输入图坐标系。

本模块复用 vision_algorithm.common.rknn_infer 的底层解码（DFL/box_process/nms），
仅实现「4 输出编排 + 关键点从第 4 路解析」这一层，不 import 业务 pose_model。
"""

import numpy as np

from vision_algorithm.common.rknn_infer import (
    sigmoid, box_process, nms, as_nchw)

NUM_KPTS = 17


# ------------------------------------------------------------------
# 关键点 flatten 输出 -> (17, 3, N)（关键点, x/y/conf, 锚点）
# ------------------------------------------------------------------
def normalize_kpts_flatten(out4):
    """把 outputs[3] 规范成 (17, 3, N) ndarray，返回 (kpts, diag)。

    支持常见导出形状：
      [1, 17, 3, N]  [1, 3, 17, N]  [1, N, 17, 3]
      [1, 51, N]  [1, N, 51]  （51 = 17×3，kpt-major 展平）
    diag 含 shape / 数值范围，供全 0 根因诊断。
    """
    arr = np.asarray(out4, dtype=np.float32)
    diag = {"raw_shape": list(arr.shape)}

    # 统一 squeeze 掉 batch=1
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        dims = list(arr.shape)
        # (17, 3, N)
        if dims[0] == NUM_KPTS and dims[1] == 3:
            k = arr
        # (3, 17, N)
        elif dims[0] == 3 and dims[1] == NUM_KPTS:
            k = np.transpose(arr, (1, 0, 2))
        # (N, 17, 3)
        elif dims[2] == 3 and dims[1] == NUM_KPTS:
            k = np.transpose(arr, (1, 2, 0))
        # (N, 3, 17)
        elif dims[2] == NUM_KPTS and dims[1] == 3:
            k = np.transpose(arr, (2, 1, 0))
        # (51, N) 展平
        elif dims[0] == NUM_KPTS * 3:
            k = arr.reshape(NUM_KPTS, 3, dims[1])
        # (N, 51) 展平
        elif dims[1] == NUM_KPTS * 3:
            k = arr.reshape(dims[0], NUM_KPTS, 3).transpose(1, 2, 0)
        else:
            k = None
            diag["error"] = f"无法识别的关键点形状 {arr.shape}"
    elif arr.ndim == 2:
        # (51, N) 或 (N, 51) 展平（无 batch）
        if arr.shape[0] == NUM_KPTS * 3:
            k = arr.reshape(NUM_KPTS, 3, arr.shape[1])
        elif arr.shape[1] == NUM_KPTS * 3:
            k = arr.reshape(arr.shape[0], NUM_KPTS, 3).transpose(1, 2, 0)
        else:
            k = None
            diag["error"] = f"无法识别的关键点形状 {arr.shape}"
    else:
        k = None
        diag["error"] = f"关键点输出维度异常 ndim={arr.ndim} shape={arr.shape}"

    if k is not None:
        # 数值范围诊断（判断是否已解码 / conf 是否已 sigmoid）
        diag["kpts_shape"] = list(k.shape)   # (17, 3, N)
        diag["x_range"] = [float(k[:, 0, :].min()), float(k[:, 0, :].max())]
        diag["y_range"] = [float(k[:, 1, :].min()), float(k[:, 1, :].max())]
        diag["conf_range"] = [float(k[:, 2, :].min()), float(k[:, 2, :].max())]
        # conf 明显未 sigmoid（>1 或 <0）时补 sigmoid
        cmin, cmax = diag["conf_range"]
        if cmax > 1.0 or cmin < 0.0:
            k[:, 2, :] = sigmoid(k[:, 2, :])
            diag["conf_sigmoid_applied"] = True
    return k, diag


# ------------------------------------------------------------------
# 关键点 conf 退化判定（全 0 -> 导出/量化丢 conf 通道）
# ------------------------------------------------------------------
def kpts_conf_degenerate(k):
    """conf 通道最大值 < 1e-3 视为退化（全 0）。"""
    return float(k[:, 2, :].max()) < 1e-3


# ------------------------------------------------------------------
# box 输出解析（outputs[0..2] 3 尺度）
# ------------------------------------------------------------------
def parse_box_outputs(outputs, model_w, model_h, conf_thres):
    """解析前 3 路 box 输出，返回 (detections, diag)。

    detections: list of dict {'box':(x1,y1,x2,y2), 'conf':float, 'scale_hw':(h,w)}
                坐标为 pose 输入图（model_w×model_h）坐标系。
    """
    dets = []
    diag = {"box_output_shapes": [], "scales": []}
    for idx, out in enumerate(outputs[:3]):
        arr = np.asarray(out)
        diag["box_output_shapes"].append(list(arr.shape))
        if arr.ndim != 4:
            diag.setdefault("warnings", []).append(
                f"box 输出[{idx}] ndim={arr.ndim} 非 4D，跳过")
            continue
        # 转 NCHW（以 64/65 通道判定布局）
        o = as_nchw(arr, None) if not (64 in arr.shape[1:] or 65 in arr.shape[1:]) \
            else as_nchw(arr, arr.shape[3] in (64, 65))
        # 注意：as_nchw 的 is_nhwc 判定用 64/65 通道在 shape[3]
        c = o.shape[1]
        h, w = o.shape[2], o.shape[3]
        if c == 65:
            box = o[:, :64, :, :]
            cls = o[:, 64:, :, :]
        elif c == 64:
            box = o
            cls = None
        elif c == 1:
            # 分离式 cls 单独一路
            diag.setdefault("warnings", []).append(
                f"box 输出[{idx}] 是 1 通道 cls 分支，需与 64 通道 box 配对")
            continue
        else:
            diag.setdefault("warnings", []).append(
                f"box 输出[{idx}] 通道数 {c} 非 64/65，跳过")
            continue

        xyxy = box_process(box, model_h, model_w)      # (1, 4, h, w)
        if cls is not None:
            conf = sigmoid(cls.astype(np.float32))[:, 0]   # (1, h, w)
        else:
            # 无独立 cls：用 box 置信度兜底（分离式可能缺失，标记）
            conf = np.ones((1, h, w), dtype=np.float32)
            diag.setdefault("warnings", []).append(
                f"box 输出[{idx}] 无 cls 通道，conf 置 1 兜底")
        mask = conf.flatten() >= conf_thres
        if not np.any(mask):
            diag["scales"].append({"hw": [h, w], "n_det": 0})
            continue
        flat_idx = np.where(mask)[0]           # 尺度内锚点序号（C 序，行优先）
        b = xyxy.reshape(4, -1).T[flat_idx]
        sc = conf.flatten()[flat_idx]
        for j in range(b.shape[0]):
            dets.append({
                'box': (float(b[j, 0]), float(b[j, 1]),
                        float(b[j, 2]), float(b[j, 3])),
                'conf': float(sc[j]),
                'scale_hw': (h, w),
                'flat_idx': int(flat_idx[j]),   # 尺度内锚点序号（用于索引 flatten 关键点）
            })
        diag["scales"].append({"hw": [h, w], "n_det": int(b.shape[0])})
    return dets, diag


# ------------------------------------------------------------------
# 4 输出总解析（复现业务 detect_crop 的关键点解析，但用 outputs[3]）
# ------------------------------------------------------------------
def parse_pose4(outputs, model_w, model_h, conf_thres, nms_thres):
    """解析 pose 4 输出，返回 (results, diag)。

    results: list of dict {'box':(x1,y1,x2,y2), 'conf':float,
                           'kpts':(17,2), 'kpt_conf':(17,)}
             box/kpts 均为 pose 输入图（model_w×model_h）坐标系。
    diag   : 诊断 dict（输出 shape、关键点形状/范围、尺度锚点数等）。
    """
    diag = {"n_outputs": len(outputs),
            "output_shapes": [list(np.asarray(o).shape) for o in outputs]}

    dets, box_diag = parse_box_outputs(outputs, model_w, model_h, conf_thres)
    diag.update(box_diag)

    # 关键点：固定取第 4 路（outputs[3]）
    kpts_flat = None
    kpts_diag = {}
    if len(outputs) < 4:
        diag["kpts_error"] = (
            f"模型仅 {len(outputs)} 路输出，缺少第 4 路 flatten 关键点输出（outputs[3]）。"
            "请确认导出为 flatten=True 的 4 输出结构。")
    else:
        kpts_flat, kpts_diag = normalize_kpts_flatten(outputs[3])
    diag.update(kpts_diag)

    if not dets or kpts_flat is None:
        return [], diag

    # 关键点锚点顺序 = 大尺度在前（与 box 输出按尺度面积降序一致）。
    # 关键：offset 必须按【所有 3 个尺度】累积（含未检出框的尺度），因为 flatten
    # 关键点里每个尺度的锚点都占位；若只按「检测到框的尺度」累积，遇到某尺度
    # 未检出框时 offset 会整体前移，导致关键点取到错误锚点（表现为 17 点全 0）。
    all_hws = []
    for s in box_diag.get("scales", []):
        hw = tuple(s["hw"])
        if hw not in all_hws:
            all_hws.append(hw)
    scale_order = sorted(all_hws, key=lambda hw: -hw[0] * hw[1])
    scale_offsets = {}
    off = 0
    for hw in scale_order:
        scale_offsets[hw] = off
        off += hw[0] * hw[1]
    total_anchors = off
    diag["scale_order"] = [list(hw) for hw in scale_order]
    diag["scale_offsets"] = {f"{hw[0]}x{hw[1]}": v for hw, v in scale_offsets.items()}
    diag["kpts_anchor_count"] = kpts_flat.shape[2] if kpts_flat.ndim == 3 else None
    diag["box_total_anchors"] = total_anchors

    results = []
    for d in dets:
        hw = d['scale_hw']
        flat_idx = d['flat_idx']
        global_idx = scale_offsets[hw] + flat_idx
        if global_idx < kpts_flat.shape[2]:
            kk = kpts_flat[:, :, global_idx]      # (17, 3)
        else:
            kk = np.zeros((NUM_KPTS, 3), dtype=np.float32)
            diag.setdefault("warnings", []).append(
                f"全局锚点索引 {global_idx} 越界（kpts 锚点数 {kpts_flat.shape[2]}）")
        results.append({'box': d['box'], 'conf': d['conf'],
                        'scale_hw': hw, 'kpts': kk})

    # NMS（按 conf 降序）
    boxes = np.array([r['box'] for r in results], dtype=np.float32)
    scores = np.array([r['conf'] for r in results], dtype=np.float32)
    keep = nms(boxes, scores, nms_thres)

    final = []
    for i in keep:
        r = results[i]
        kk = r['kpts']
        conf = kk[:, 2].copy()
        # conf 退化（全 0）时退化为坐标判定可见性
        if kpts_conf_degenerate(kpts_flat):
            visible = (kk[:, 0] > 0) & (kk[:, 1] > 0) & \
                      (kk[:, 0] < model_w) & (kk[:, 1] < model_h)
            conf = np.where(visible, 1.0, 0.0).astype(np.float32)
        final.append({
            'box': r['box'], 'conf': r['conf'],
            'kpts': kk[:, :2], 'kpt_conf': conf,
        })
    return final, diag


# ------------------------------------------------------------------
# 关键点反 letterbox（pose 输入图坐标 -> crop 图坐标）
# ------------------------------------------------------------------
def kpts_to_crop(kpts_input, scale, dw, dh):
    """pose 输入图坐标 -> crop 图坐标：kpts_input 为 (17,3) 或 (17,2)。"""
    k = kpts_input.copy()
    k[:, 0] = (k[:, 0] - dw) / scale
    k[:, 1] = (k[:, 1] - dh) / scale
    return k


def box_to_crop(box_input, scale, dw, dh):
    """pose 输入图坐标 -> crop 图坐标。"""
    x1, y1, x2, y2 = box_input
    return ((x1 - dw) / scale, (y1 - dh) / scale,
            (x2 - dw) / scale, (y2 - dh) / scale)
