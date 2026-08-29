# -*- coding: utf-8 -*-
"""
检测模型模块（detection/det_model）
======================================
职责：目标检测（YOLOv8 两类别模型：cls=0=player，cls=1=basketball）的 RKNN 推理。
     player 框后续被裁剪出来单独喂给姿态模型，basketball 框用于持球/出手判定。

对外暴露：RKNNDetModel
依赖：vision_algorithm.common.rknn_infer（DFL 解码 / NMS / letterbox / _RKNNBase）
"""

import logging

import numpy as np

from vision_algorithm.common.rknn_infer import (
    _RKNNBase, box_process, nms, as_nchw, preprocess_to_input, canvas_to_input)

logger = logging.getLogger("basketball_scoring")


class RKNNDetModel(_RKNNBase):
    """
    YOLOv8 检测（两类别：cls=0=player，cls=1=basketball）。

    detect(frame) / detect_on_canvas(canvas) -> list of dict:
        {'box': (x1,y1,x2,y2), 'cls': int, 'conf': float}
    detect(frame) 返回 frame 原始坐标系；detect_on_canvas 返回 640x640 画布坐标系。
    """

    def __init__(self, rknn_path, conf_thres=0.45, nms_thres=0.45,
                 ball_conf_thres=0.30, model_w=640, model_h=640,
                 core_mask=None):
        super().__init__(rknn_path, model_w=model_w, model_h=model_h,
                         core_mask=core_mask)
        self.conf_thres = conf_thres
        self.ball_conf_thres = ball_conf_thres  # 篮球单独阈值（小目标更宽容）
        self.nms_thres = nms_thres
        self.num_classes = None  # 首次推理时从 cls 分支通道数确定

    def _group_outputs(self, outputs):
        """
        把检测模型输出按「空间分辨率」分组，返回每尺度 (box_dfl, cls, obj) 三元组。

        兼容多种 RKNN/ONNX 导出结构：
          A. 分离式 9 输出：box(64) + cls(nc) + objectness/score_sum(1)
             （obj 与 cls 相乘得最终置信度，见 _postprocess）
          B. 合并式：box(64) + cls(nc) 拼接为单路输出（65=单类, 66=两类，无 obj）
        单类别模型（只检测篮球）时 cls 与 objectness 都是 1 通道且数值相同，此时
        该 1 通道同时充当 cls（obj 缺失则不乘）。
        """
        from collections import defaultdict

        # 1) 判定布局：box(64) 或合并输出(65/66) 的宽通道维度所在位置
        is_nhwc = None
        for out in outputs:
            if out.ndim != 4:
                continue
            if out.shape[3] in (64, 65, 66):
                is_nhwc = True
                break
            if out.shape[1] in (64, 65, 66):
                is_nhwc = False
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
            elif c > 64:
                # 合并式：box(64) + cls(c-64) 拼接
                by_scale[hw]['box'] = out[:, :64, :, :]
                by_scale[hw]['cls'] = out[:, 64:, :, :]
                if self.num_classes is None:
                    self.num_classes = c - 64
            elif c > 1:
                # 分离式多类别 cls 分支
                by_scale[hw]['cls'] = out
                if self.num_classes is None:
                    self.num_classes = c
            elif c == 1:
                # 1 通道：objectness/score_sum（分离式）。若该尺度无独立 cls，则它本身即单类 cls。
                by_scale[hw].setdefault('_c1', []).append(out)

        # 3) 处理 1 通道分支：
        #    该尺度已有独立 cls(>1通道) 时，1 通道是 objectness/score_sum，存为 obj；
        #    否则（单类别模型）首个 1 通道即 cls。
        for d in by_scale.values():
            if 'cls' in d:
                d['obj'] = d['_c1'][0] if '_c1' in d else None
                d.pop('_c1', None)
                continue
            if '_c1' in d:
                d['cls'] = d['_c1'][0]
                d['obj'] = None
                d.pop('_c1', None)
                if self.num_classes is None:
                    self.num_classes = 1

        # 4) 一次性诊断日志：记录真实输出结构与类别判定（便于排查「识别不到目标」）
        if not getattr(self, "_logged_outputs", False):
            shapes = [tuple(o.shape) for o in outputs if o.ndim == 4]
            logger.info(
                "检测模型输出结构诊断: 4D 输出形状=%s | 判定类别数=%s",
                shapes, self.num_classes)
            self._logged_outputs = True

        groups = []
        for hw in sorted(by_scale, key=lambda k: -k[0] * k[1]):
            d = by_scale[hw]
            if 'box' in d and 'cls' in d:
                groups.append((d['box'], d['cls'], d.get('obj')))
        return groups

    def detect(self, frame):
        img, scale, dw, dh = preprocess_to_input(frame, self.model_w, self.model_h)
        return self._detect_from_input(img, scale, dw, dh)

    def detect_on_canvas(self, canvas):
        """对已 letterbox 的 640x640 BGR 画布做检测，返回「画布坐标系」的检测结果。

        这是主推理入口：调用方先 letterbox 得到画布（描黑边，省 YOLO 内部再预处理），
        再调用本方法，拿到的 box 坐标即 640x640 画布坐标，便于裁剪 player / 反算原图。
        """
        img = canvas_to_input(canvas)
        outputs = self.infer_input(img)
        return self._postprocess(outputs)

    def _detect_from_input(self, img, scale, dw, dh):
        """对已 letterbox 的输入做推理 + 后处理，并把坐标反算回原始帧坐标系。"""
        outputs = self.infer_input(img)
        results = self._postprocess(outputs)

        # 模型输入(画布)坐标 -> 原始帧坐标
        mapped = []
        for r in results:
            bx1, by1, bx2, by2 = r['box']
            mapped.append({
                'box': (int((bx1 - dw) / scale), int((by1 - dh) / scale),
                        int((bx2 - dw) / scale), int((by2 - dh) / scale)),
                'cls': r['cls'],
                'conf': r['conf'],
            })
        return mapped

    def _postprocess(self, outputs):
        """推理原始输出 -> 检测结果列表（坐标为模型输入 640x640 画布坐标系）。"""
        groups = self._group_outputs(outputs)

        all_boxes, all_scores, all_classes = [], [], []
        for box_out, cls_out, obj_out in groups:
            xyxy = box_process(box_out, self.model_h, self.model_w)  # (1,4,h,w)
            # cls 分支在 ONNX 导出时已做过 sigmoid（输出名即 'sigmoid'），勿二次 sigmoid
            cls_conf = cls_out.astype(np.float32)                    # (1,nc,h,w) 已 [0,1]
            max_conf = cls_conf.max(1)                               # (1,h,w)
            max_cls = cls_out.argmax(1)                              # (1,h,w)

            # 分离式 9 输出带 objectness/score_sum(1通道)：与 max_cls 相乘得最终置信度
            # （参考 test/detect_video.py 的 [box_reg, cls, obj] 口径）。合并式无 obj 则跳过。
            if obj_out is not None:
                obj = obj_out.astype(np.float32)[:, 0]               # (1,h,w) 已 [0,1]
                score = max_conf * obj
            else:
                score = max_conf

            # 类别自适应阈值（作用于 obj×cls 后的置信度）：
            #   两类别模型：player(cls=0) 用 conf_thres，basketball(cls=1) 用更低阈值
            #   （篮球是小目标，单独更宽容，避免漏检）。
            if cls_out.shape[1] == 1:
                thres = np.full(score.shape, self.ball_conf_thres, dtype=np.float32)
            else:
                thres = np.full(score.shape, self.conf_thres, dtype=np.float32)
                if self.ball_conf_thres is not None:
                    thres = np.where(max_cls == 1,
                                     np.float32(self.ball_conf_thres), thres)

            mask = score.flatten() >= thres.flatten()
            if not np.any(mask):
                continue
            b = xyxy.reshape(4, -1).T[mask]                          # (N,4)
            all_boxes.append(b)
            all_scores.append(score.flatten()[mask])
            all_classes.append(max_cls.flatten()[mask])

        if not all_boxes:
            # 节流诊断：每 30 次推理打印一次，便于定位「目标检不出」
            self._det_log_count = getattr(self, "_det_log_count", 0) + 1
            if self._det_log_count % 30 == 1:
                logger.info("检测诊断: 本帧未检出任何目标（候选框为空，阈值=%.2f）",
                            self.ball_conf_thres)
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

        # 画布(640x640)坐标系结果，坐标反算由调用方按需完成
        results = []
        for i in keep_total:
            results.append({
                'box': (int(boxes[i, 0]), int(boxes[i, 1]),
                        int(boxes[i, 2]), int(boxes[i, 3])),
                'cls': int(classes[i]),
                'conf': float(scores[i]),
            })
        # 按置信度降序，调用方取首个即最优目标
        results.sort(key=lambda r: -r['conf'])

        # 节流诊断：每 30 次推理打印一次检出结果（类别/置信度）
        self._det_log_count = getattr(self, "_det_log_count", 0) + 1
        if self._det_log_count % 30 == 1:
            logger.info("检测诊断: 本次检出=%d 个目标 %s",
                        len(results),
                        [(r['cls'], round(r['conf'], 3)) for r in results])
        return results
