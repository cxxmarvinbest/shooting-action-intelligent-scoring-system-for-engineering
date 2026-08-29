import cv2
import numpy as np
from rknn.api import RKNN

# =====================配置区====================
RKNN_MODEL = "./best_fp.rknn"
VIDEO_PATH = "./left_side_basketball.mp4"
OUT_VIDEO = "./ball_result.mp4"
IMG_SIZE = 640
CONF_THRESH = 0.05
NMS_THRESH = 0.6
# 类别映射：0运动员，1篮球
CLASS_NAMES = {0: "player", 1: "basketball"}

STRIDES = [8, 16, 32]
# 9输出映射 [box_reg, cls, obj]
OUTPUT_MAP = [
    [0, 1, 2],   # stride8  80x80
    [3, 4, 5],   # stride16 40x40
    [6, 7, 8],   # stride32 20x20
]
# ==============================================

def generate_grids(grid_w, grid_h):
    yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    grid = np.stack((xv, yv), axis=-1).astype(np.float32)
    return grid

def dfl_decode(feat):
    """feat (64, gh, gw) → (gh,gw,4) l t r b"""
    feat = np.transpose(feat,(1,2,0))
    bs = np.split(feat, 4, axis=-1)
    dist = np.arange(16, dtype=np.float32)
    out = []
    for b in bs:
        b_max = np.max(b, axis=-1, keepdims=True)
        exp_b = np.exp(b - b_max)
        soft = exp_b / np.sum(exp_b,axis=-1,keepdims=True)
        val = np.sum(soft * dist, axis=-1)
        out.append(val)
    return np.stack(out,axis=-1)


def postprocess_yolov8_9output(output_list, conf_thresh, nms_thresh, model_size, orig_w, orig_h):
    boxes_all = []
    scores_all = []
    cls_ids_all = []

    for level_idx, stride in enumerate(STRIDES):
        box_idx, cls_idx, obj_idx = OUTPUT_MAP[level_idx]
        box_feat = output_list[box_idx][0]
        cls_feat = output_list[cls_idx][0]
        obj_feat = output_list[obj_idx][0]

        _, gh, gw = box_feat.shape
        grid = generate_grids(gw, gh)
        box_ltrb = dfl_decode(box_feat)

        cls_feat = np.transpose(cls_feat,(1,2,0))
        obj_feat = np.transpose(obj_feat,(1,2,0))
        obj_feat = np.squeeze(obj_feat,axis=-1)

        for yi in range(gh):
            for xi in range(gw):
                l, t, r, b = box_ltrb[yi, xi]
                obj_conf = obj_feat[yi, xi]
                cls_scores = cls_feat[yi, xi]

                max_cls_score = np.max(cls_scores)
                max_score = float(obj_conf * max_cls_score)
                if max_score < conf_thresh:
                    continue
                cls_id = int(np.argmax(cls_scores))

                gx, gy = grid[yi, xi]
                # DFL ltrb得到模型640尺度坐标
                x1 = (gx - l) * stride
                y1 = (gy - t) * stride
                x2 = (gx + r) * stride
                y2 = (gy + b) * stride

                # 映射回原图分辨率
                scale = orig_w / model_size
                x1 = x1 * scale
                x2 = x2 * scale
                scale_h = orig_h / model_size
                y1 = y1 * scale_h
                y2 = y2 * scale_h

                # 边界限制
                x1 = np.clip(x1, 0, orig_w)
                y1 = np.clip(y1, 0, orig_h)
                x2 = np.clip(x2, 0, orig_w)
                y2 = np.clip(y2, 0, orig_h)

                boxes_all.append([float(x1), float(y1), float(x2), float(y2)])
                scores_all.append(max_score)
                cls_ids_all.append(cls_id)

    if len(boxes_all) == 0:
        return [], [], []

    indices = cv2.dnn.NMSBoxes(boxes_all, scores_all, conf_thresh, nms_thresh)
    indices = np.array(indices).flatten()

    final_boxes  = [boxes_all[i] for i in indices]
    final_scores = [scores_all[i] for i in indices]
    final_cls    = [cls_ids_all[i] for i in indices]
    return final_boxes, final_scores, final_cls


def main():
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("加载rknn失败")
        return
    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        print("init runtime error")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (orig_w, orig_h))

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_src = frame.copy()

        # 预处理：BGR→RGB resize→float32→NCHW
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        input_data = resized.astype(np.float32)
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        outputs = rknn.inference(inputs=[input_data], data_format="nchw")
        boxes, scores, cls_ids = postprocess_yolov8_9output(
            outputs, CONF_THRESH, NMS_THRESH, IMG_SIZE, orig_w, orig_h
        )

        count_player = sum(1 for c in cls_ids if c == 0)
        count_ball = sum(1 for c in cls_ids if c == 1)
        print(f"帧{frame_id:4d} | 人:{count_player} 球:{count_ball}")

        for box, score, cid in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            cls_name = CLASS_NAMES[cid]
            if cid == 0:
                color = (0, 255, 0)   # player绿色
            else:
                color = (0, 0, 255)   # basketball红色
            cv2.rectangle(frame_src, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_src, f"{cls_name} {score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        out.write(frame_src)
        frame_id += 1

    cap.release()
    out.release()
    rknn.release()
    print(f"\n处理完成，输出视频保存至 {OUT_VIDEO}")


if __name__ == "__main__":
    main()
