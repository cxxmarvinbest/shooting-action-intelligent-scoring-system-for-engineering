# -*- coding: utf-8 -*-
"""
姿态模型模块（pose_estimate/pose_model）
==========================================
职责：人体姿态估计（yolov8-pose，17 个 COCO 关键点）的 RKNN 推理。

对外暴露：RKNNPoseModel
依赖：common.rknn_utils（DFL 解码 / 关键点解码 / NMS / letterbox / _RKNNBase）
"""

import numpy as np

from common.rknn_utils import (
    _RKNNBase, sigmoid, box_process, nms, as_nchw)


class RKNNPoseModel(_RKNNBase):
    """
    YOLOv8 姿态估计（单人/多人，17 个 COCO 关键点）。

    detect(frame) -> list of dict:
        {'box': (x1,y1,x2,y2), 'conf': float, 'kpts': (17,2) ndarray}
    kpts 为输入 frame 原始坐标；不可见点坐标为 0（与原 ultralytics 版口径一致）。
    """

    NUM_KPTS = 17

    def __init__(self, rknn_path, conf_thres=0.3, nms_thres=0.45,
                 kpt_conf_thres=0.5, model_w=640, model_h=640):
        super().__init__(rknn_path, model_w=model_w, model_h=model_h)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.kpt_conf_thres = kpt_conf_thres

    def _group_outputs(self, outputs):
        """
        把姿态模型 4 路输出拆分为「每尺度 (box_dfl, cls)」+「已解码关键点」。

        yolov8n-pose.onnx 实际输出：
          - 3 × [1,65,h,w]   ：box(64) + cls(1) 合并输出（raw，需 DFL 解码 + sigmoid）
          - 1 × [1,17,3,8400]：已解码关键点 (x, y, conf)，x/y 为输入图坐标、conf 已 sigmoid
        返回 (pairs, kpts)；pairs 按尺度降序（80→40→20），与 kpts 的 8400 锚点顺序一致。
        """
        from collections import defaultdict

        is_nhwc = None
        kpts = None
        # 1) 提取已解码关键点 + 用合并输出(65 通道)判定布局
        for out in outputs:
            if out.ndim != 4:
                continue
            if 8400 in out.shape[1:] and 17 in out.shape[1:]:
                kpts = self._normalize_kpts(out)
            elif (65 in out.shape[1:] or 64 in out.shape[1:]) and is_nhwc is None:
                is_nhwc = (out.shape[3] in (64, 65))

        # 2) 统一转 NCHW，按 (h, w) 分组
        by_scale = defaultdict(dict)
        for out in outputs:
            if out.ndim != 4:
                continue
            if 8400 in out.shape[1:] and 17 in out.shape[1:]:
                continue                        # kpts 已单独处理
            out = as_nchw(out, is_nhwc)         # (1, C, h, w)
            c = out.shape[1]
            hw = (out.shape[2], out.shape[3])
            if c == 65:
                by_scale[hw]['box'] = out[:, :64, :, :]
                by_scale[hw]['cls'] = out[:, 64:, :, :]
            elif c == 64:
                by_scale[hw]['box'] = out
            elif c == 1:
                by_scale[hw]['cls'] = out

        pairs = []
        for hw in sorted(by_scale, key=lambda k: -k[0] * k[1]):
            d = by_scale[hw]
            if 'box' in d and 'cls' in d:
                pairs.append((d['box'], d['cls']))
        return pairs, kpts

    @staticmethod
    def _normalize_kpts(out):
        """把已解码关键点张量规范为 (1, 17, 3, 8400)（batch, 关键点, x/y/conf, 锚点）。"""
        if out.ndim != 4:
            return out
        if out.shape[1:] == (17, 3, 8400):
            return out
        dims = list(out.shape[1:])
        if 8400 not in dims or 17 not in dims:
            return out
        ax_anchor = dims.index(8400)
        rest = [i for i in range(3) if i != ax_anchor]
        ax_k = rest[0] if dims[rest[0]] == 17 else rest[1]
        ax_c = rest[1] if ax_k == rest[0] else rest[0]
        return np.transpose(out, (0, ax_k + 1, ax_c + 1, ax_anchor + 1))

    def detect(self, frame):
        outputs, scale, dw, dh = self.infer(frame)
        pairs, kpts = self._group_outputs(outputs)

        all_boxes, all_scores, all_kpts = [], [], []
        offset = 0  # 当前尺度在 8400 个全尺度锚点中的起始序号
        for box_out, cls_out in pairs:
            xyxy = box_process(box_out, self.model_h, self.model_w)   # (1,4,h,w)
            conf = sigmoid(cls_out.astype(np.float32))[:, 0]          # (1,h,w) 仅 person 类
            mask = conf.flatten() >= self.conf_thres
            n_cell = conf.size
            if not np.any(mask):
                offset += n_cell
                continue
            flat_idx = np.where(mask)[0]          # 本尺度内锚点序号
            all_boxes.append(xyxy.reshape(4, -1).T[flat_idx])
            all_scores.append(conf.flatten()[flat_idx])
            if kpts is not None:
                # 已解码关键点：按全局锚点序号取 (17,3,N) -> (N,17,3)
                k = kpts[0][:, :, offset + flat_idx]
                all_kpts.append(k.transpose(2, 0, 1))
            offset += n_cell

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        if all_kpts:
            kpts_all = np.concatenate(all_kpts)
        else:
            kpts_all = np.zeros((boxes.shape[0], self.NUM_KPTS, 3), dtype=np.float32)

        keep = nms(boxes, scores, self.nms_thres)

        results = []
        fh, fw = frame.shape[:2]
        for i in keep:
            x1 = (boxes[i, 0] - dw) / scale
            y1 = (boxes[i, 1] - dh) / scale
            x2 = (boxes[i, 2] - dw) / scale
            y2 = (boxes[i, 3] - dh) / scale
            kpts = kpts_all[i].copy()                 # (17,3)
            kpts[:, 0] = (kpts[:, 0] - dw) / scale
            kpts[:, 1] = (kpts[:, 1] - dh) / scale
            # 低置信关键点坐标置 0，保持与原 ultralytics 版一致的口径
            low_conf = kpts[:, 2] < self.kpt_conf_thres
            kpts[low_conf, 0] = 0.0
            kpts[low_conf, 1] = 0.0
            # 越界保护
            kpts[:, 0] = np.clip(kpts[:, 0], 0, fw - 1)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, fh - 1)
            results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'conf': float(scores[i]),
                'kpts': kpts[:, :2],
            })
        return results
