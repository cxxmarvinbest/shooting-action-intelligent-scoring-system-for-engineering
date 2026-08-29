import cv2
import numpy as np

class YOLOv8Postprocess:
    def __init__(self, conf_threshold=0.25, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def postprocess(self, outputs, img_shape, model_input_size):
        """
        outputs: rknn推理输出
        img_shape: (h,w,c)原图尺寸
        model_input_size: 640
        return boxes, classes, scores
        """
        boxes = []
        classes = []
        scores = []

        output = outputs[0].transpose((0, 2, 1))
        batch_size, num_dets, output_dim = output.shape
        orig_h, orig_w = img_shape[:2]

        for bi in range(batch_size):
            pred = output[bi]
            for det in pred:
                cls_conf = det[4:]
                max_conf = np.max(cls_conf)
                if max_conf < self.conf_threshold:
                    continue
                cls_id = np.argmax(cls_conf)
                cx, cy, w, h = det[0], det[1], det[2], det[3]

                # 映射回原图坐标
                scale_w = orig_w / model_input_size
                scale_h = orig_h / model_input_size

                x1 = int((cx - w / 2) * scale_w)
                y1 = int((cy - h / 2) * scale_h)
                x2 = int((cx + w / 2) * scale_w)
                y2 = int((cy + h / 2) * scale_h)

                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)
                scores.append(float(max_conf))

        # NMS
        if len(boxes) > 0:
            boxes_np = np.array(boxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(
                boxes_np.tolist(), scores_np.tolist(),
                self.conf_threshold, self.nms_threshold
            )
            out_boxes = [boxes[i] for i in indices]
            out_cls = [classes[i] for i in indices]
            out_scores = [scores[i] for i in indices]
            return out_boxes, out_cls, out_scores
        else:
            return [], [], []
