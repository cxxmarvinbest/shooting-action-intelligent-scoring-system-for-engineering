# -*- coding: utf-8 -*-
"""
检测模型模块（tracker/det_model）
==================================
职责：球员/篮球检测（自训练 YOLOv8 模型，类别 0=player, 1=ball）的 RKNN 推理。

对外暴露：RKNNDetModel
依赖：common.rknn_utils（DFL 解码 / NMS / letterbox / _RKNNBase）
"""

import numpy as np

from common.rknn_utils import (
    _RKNNBase, box_process, nms, as_nchw)


class RKNNDetModel(_RKNNBase):
    """
    YOLOv8 检测（自训练：0=player, 1=ball）。

    detect(frame) -> list of dict:
        {'box': (x1,y1,x2,y2), 'cls': int, 'conf': float}
    坐标为输入 frame 的原始坐标系。
    """

    def __init__(self, rknn_path, conf_thres=0.45, nms_thres=0.45,
                 model_w=640, model_h=640):
        super().__init__(rknn_path, model_w=model_w, model_h=model_h)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.num_classes = None  # 首次推理时从 cls 分支通道数确定

    def _group_outputs(self, outputs):
        """
        把检测模型 9 路输出按「空间分辨率」分组，返回每尺度 (box_dfl, cls) 对。

        best.onnx 实际输出（3 尺度 × 3 分支）：
          box_dfl(64) + cls(2, 已 sigmoid) + objectness(1)  —— 共 9 路。
        1 通道的 objectness 分支是冗余的（= cls 各通道之和），直接忽略。
        按 h*w 分组可避免旧版「分别排序再 zip」导致的跨尺度错配。
        """
        from collections import defaultdict

        # 1) 用 box 分支（64 通道）判定 rknn 输出布局（NHWC 还是 NCHW）
        is_nhwc = None
        for out in outputs:
            if out.ndim == 4 and 64 in out.shape[1:]:
                is_nhwc = (out.shape[3] == 64)
                break

        # 2) 统一转 NCHW，按 (h, w) 分组
        by_scale = defaultdict(dict)
        for out in outputs:
            if out.ndim != 4:
                continue
            out = as_nchw(out, is_nhwc)          # (1, C, h, w)
            c = out.shape[1]
            hw = (out.shape[2], out.shape[3])
            if c == 64:
                by_scale[hw]['box'] = out
            elif c > 1:
                by_scale[hw]['cls'] = out
                self.num_classes = c
            # c == 1：objectness 分支，忽略

        pairs = []
        for hw in sorted(by_scale, key=lambda k: -k[0] * k[1]):
            d = by_scale[hw]
            if 'box' in d and 'cls' in d:
                pairs.append((d['box'], d['cls']))
        return pairs

    def detect(self, frame):
        outputs, scale, dw, dh = self.infer(frame)
        pairs = self._group_outputs(outputs)

        all_boxes, all_scores, all_classes = [], [], []
        for box_out, cls_out in pairs:
            xyxy = box_process(box_out, self.model_h, self.model_w)  # (1,4,h,w)
            # cls 分支在 ONNX 导出时已做过 sigmoid（输出名即 'sigmoid'），勿二次 sigmoid
            cls_conf = cls_out.astype(np.float32)                    # (1,nc,h,w) 已 [0,1]
            max_conf = cls_conf.max(1)                               # (1,h,w)
            max_cls = cls_out.argmax(1)                              # (1,h,w)

            mask = max_conf.flatten() >= self.conf_thres
            if not np.any(mask):
                continue
            b = xyxy.reshape(4, -1).T[mask]                          # (N,4)
            all_boxes.append(b)
            all_scores.append(max_conf.flatten()[mask])
            all_classes.append(max_cls.flatten()[mask])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)

        # 按类别做 NMS
        keep_total = []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            keep = nms(boxes[idx], scores[idx], self.nms_thres)
            keep_total.extend(idx[keep].tolist())

        # 模型输入坐标 -> 原始帧坐标
        results = []
        for i in keep_total:
            x1 = (boxes[i, 0] - dw) / scale
            y1 = (boxes[i, 1] - dh) / scale
            x2 = (boxes[i, 2] - dw) / scale
            y2 = (boxes[i, 3] - dh) / scale
            results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'cls': int(classes[i]),
                'conf': float(scores[i]),
            })
        # 按置信度降序，调用方取首个即最优目标
        results.sort(key=lambda r: -r['conf'])
        return results
