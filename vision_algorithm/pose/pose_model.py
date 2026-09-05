# -*- coding: utf-8 -*-
"""
姿态模型模块（pose/pose_model）
==================================
职责：人体姿态估计（yolov8-pose，17 个 COCO 关键点）的 RKNN 推理。

对外暴露：RKNNPoseModel
依赖：vision_algorithm.common.rknn_infer（DFL 解码 / 关键点解码 / NMS / letterbox / _RKNNBase）
"""

import logging

import numpy as np

from vision_algorithm.common.rknn_infer import (
    _RKNNBase, sigmoid, box_process, nms, as_nchw, kps_process,
    preprocess_to_input)

logger = logging.getLogger("basketball_scoring")


class RKNNPoseModel(_RKNNBase):
    """
    YOLOv8 姿态估计（17 个 COCO 关键点），固定静态输入 320(W)×320(H)（NCHW [1,3,320,320]）。

    detect_crop(crop_bgr) -> list of dict:
        {'box': (x1,y1,x2,y2), 'conf': float, 'kpts': (17,2) ndarray}
    kpts 为裁剪图坐标系；不可见点坐标为 0（与原 ultralytics 版口径一致）。
    """

    NUM_KPTS = 17

    def __init__(self, rknn_path, *, conf_thres, nms_thres, kpt_conf_thres,
                 model_w, model_h, core_mask=None):
        super().__init__(rknn_path, model_w=model_w, model_h=model_h,
                         core_mask=core_mask)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.kpt_conf_thres = kpt_conf_thres
        # 关键点 conf 通道是否退化（全 0）。部分 INT8 量化/ONNX 导出会丢关键点置信度，
        # 导致 x/y 正常但 conf 全 0，此时改用「坐标判定可见性」兜底。
        self._conf_degenerate = False

    def _group_outputs(self, outputs):
        """
        把姿态模型输出拆分为「每尺度 (box_dfl, cls, kpts_raw)」+「可选已解码关键点」。

        兼容两种导出结构：
          A. 已解码关键点：1 × [1,17,3,N]（x/y/conf 已解码，conf 已 sigmoid）
          B. 原始关键点：每尺度 1 × [1,51,h,w]（51=17*3 的 raw logits，需 kps_process 解码）
        两种都返回 (pairs, kpts_decoded)；kpts_decoded 为 None 时说明走 B（raw），
        每尺度原始关键点放在 pairs 的第三元（kpts_raw）。
        """
        from collections import defaultdict

        is_nhwc = None
        kpts_decoded = None
        # 1) 用合并输出(65 通道)或 box(64)判定布局
        for out in outputs:
            if out.ndim != 4:
                continue
            if (65 in out.shape[1:] or 64 in out.shape[1:]) and is_nhwc is None:
                is_nhwc = (out.shape[3] in (64, 65))

        # 2) 统一转 NCHW，按 (h, w) 分组
        by_scale = defaultdict(dict)
        for out in outputs:
            if out.ndim != 4:
                continue
            # 已解码关键点：形状同时含 17 与 3
            if 17 in out.shape[1:] and 3 in out.shape[1:]:
                kpts_decoded = self._normalize_kpts(out)
                continue
            out = as_nchw(out, is_nhwc)         # (1, C, h, w)
            c = out.shape[1]
            hw = (out.shape[2], out.shape[3])
            if c == 65:
                by_scale[hw]['box'] = out[:, :64, :, :]
                by_scale[hw]['cls'] = out[:, 64:, :, :]
            elif c == 64:
                by_scale[hw]['box'] = out
            elif c == 51:
                by_scale[hw]['kpts'] = out      # 原始关键点 logits
            elif c == 1:
                by_scale[hw]['cls'] = out

        # 3) 一次性诊断：记录真实输出结构与关键点格式（便于排查「关键点全 0」）
        if not getattr(self, "_logged_outputs", False):
            shapes = [tuple(o.shape) for o in outputs if o.ndim == 4]
            logger.info(
                "姿态模型输出结构诊断: 4D 输出形状=%s | 已解码kpts=%s",
                shapes, kpts_decoded is not None)
            self._logged_outputs = True
            # 额外诊断：关键点 conf 通道是否退化（全 0 → 导出/量化丢了关键点置信度）
            if kpts_decoded is not None:
                max_conf = float(np.asarray(kpts_decoded)[0, :, 2, :].max())
                self._conf_degenerate = max_conf < 1e-3
                logger.info(
                    "姿态关键点置信度诊断: conf 通道最大值=%.6f | 判定为退化=%s",
                    max_conf, self._conf_degenerate)

        pairs = []
        for hw in sorted(by_scale, key=lambda k: -k[0] * k[1]):
            d = by_scale[hw]
            if 'box' in d and 'cls' in d:
                pairs.append((d['box'], d['cls'], d.get('kpts')))
        return pairs, kpts_decoded

    @staticmethod
    def _normalize_kpts(out):
        """把已解码关键点张量规范为 (1, 17, 3, N)（batch, 关键点, x/y/conf, 锚点）。

        锚点数 N 随输入尺寸变化（640x640 -> 8400；动态小图 -> 更少），
        这里不写死 8400，而是按「17/3 之外的维度即锚点维」统一转置。
        """
        if out.ndim != 4:
            return out
        if out.shape[1] == 17 and out.shape[2] == 3:
            return out                      # 已是 (1, 17, 3, N)
        dims = list(out.shape[1:])
        if 17 not in dims or 3 not in dims:
            return out
        ax_k = dims.index(17)
        ax_c = dims.index(3)
        ax_n = [i for i in range(3) if i not in (ax_k, ax_c)][0]
        return np.transpose(out, (0, ax_k + 1, ax_c + 1, ax_n + 1))

    def detect(self, frame):
        img, scale, dw, dh = preprocess_to_input(frame, self.model_w, self.model_h)
        return self._detect_from_input(img, scale, dw, dh, frame.shape[:2])

    def detect_crop(self, crop_bgr):
        """对 player 裁剪图（BGR, uint8，从原图抠出的 ROI）做姿态估计。

        固定输入 320×320：采用「保持宽高比 + 黑边填充」letterbox（**不暴力拉伸/缩放**，
        避免人体变形影响关键点精度），推理后关键点坐标经反 letterbox 回裁剪图坐标系返回
        （调用方再按裁剪偏移映射回原图）。

        返回：list of dict，同 detect()，但 box/kpts 坐标为裁剪图坐标系。
        """
        # 保持宽高比 letterbox 到固定 320×320，黑边填充；_detect_from_input 会自动按
        # scale/dw/dh 把框和关键点反算回裁剪图坐标系（复用检测同一套 letterbox 逻辑）。
        img, scale, dw, dh = preprocess_to_input(crop_bgr, self.model_w, self.model_h)
        return self._detect_from_input(img, scale, dw, dh, crop_bgr.shape[:2])

    def _detect_from_input(self, img, scale, dw, dh, fh_fw):
        """对已 letterbox 的输入做推理 + 后处理（供与检测模型共享 letterbox 输入时复用）。"""
        outputs = self.infer_input(img)
        pairs, kpts_decoded = self._group_outputs(outputs)

        all_boxes, all_scores, all_kpts = [], [], []
        offset = 0  # 当前尺度在全尺度锚点中的起始序号（仅已解码 kpts 用）
        for box_out, cls_out, kpts_raw in pairs:
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
            if kpts_decoded is not None:
                # 已解码关键点：按全局锚点序号取 (17,3,N) -> (N,17,3)
                k = kpts_decoded[0][:, :, offset + flat_idx]
                all_kpts.append(k.transpose(2, 0, 1))
            elif kpts_raw is not None:
                # 原始关键点 logits (1,51,h,w) -> kps_process 解码为 (1,17,3,h,w) -> 展平取本尺度锚点
                kd = kps_process(kpts_raw, self.model_h, self.model_w)  # (1,17,3,h,w)
                k = kd[0].reshape(self.NUM_KPTS, 3, n_cell)[:, :, flat_idx]  # (17,3,Nsel)
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
        fh, fw = fh_fw
        for i in keep:
            x1 = (boxes[i, 0] - dw) / scale
            y1 = (boxes[i, 1] - dh) / scale
            x2 = (boxes[i, 2] - dw) / scale
            y2 = (boxes[i, 3] - dh) / scale
            kpts = kpts_all[i].copy()                 # (17,3)
            kpts[:, 0] = (kpts[:, 0] - dw) / scale
            kpts[:, 1] = (kpts[:, 1] - dh) / scale
            conf = kpts[:, 2]
            if self._conf_degenerate:
                # conf 通道退化（全 0）：退化为坐标判定可见性（x/y 在裁剪图内即视为可见）
                visible = (kpts[:, 0] > 0) & (kpts[:, 1] > 0) & \
                          (kpts[:, 0] < fw) & (kpts[:, 1] < fh)
                kpts[~visible, 0] = 0.0
                kpts[~visible, 1] = 0.0
                conf = np.where(visible, 1.0, 0.0).astype(np.float32)
            else:
                # 低置信关键点坐标置 0，保持与原 ultralytics 版一致的口径
                low_conf = conf < self.kpt_conf_thres
                kpts[low_conf, 0] = 0.0
                kpts[low_conf, 1] = 0.0
            # 越界保护
            kpts[:, 0] = np.clip(kpts[:, 0], 0, fw - 1)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, fh - 1)
            results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'conf': float(scores[i]),
                'kpts': kpts[:, :2],
                'kpt_conf': conf.copy(),  # 各关键点置信度（低置信点坐标已置 0）
            })
        return results
