# -*- coding: utf-8 -*-
"""
单帧「检测 + 姿态」耗时测试（test/test_speed_det_pose）
=============================================================
目的：实测实时链路下，单帧「预处理 / 检测 / 姿态」各阶段耗时与 FPS，
     用于确定 FRAME_STRIDE 取值 + anti-flicker 的 miss_max 参数。

计时口径（与真实实时链路一致，串行 检测->抠 ROI->姿态）：
  - 检测引擎由 DET_ENGINE 决定（自动跟随 video_analyzer.load_models）：
      rknn_lite：预处理=copyMakeBorder 补黑边 640x640（Python CPU）；检测=detect_on_canvas
      cpp      ：无 Python 预处理（C++ 内部 letterbox）；检测=detect_360
  - 姿态  ：detect_crop（含 letterbox 320 + NPU 推理 + 关键点后处理）

说明：
  - det360 / preview 的 resize 模拟 RGA 硬件输出，不计入计时（真实链路 RGA 0 CPU）。
  - 抠 ROI（numpy 切片）耗时极短（<0.1ms），归入姿态耗时内。
  - 切 DET_ENGINE=cpp 后本脚本自动走 cpp 分支，无需改动。

用法（RK3588 板端）：
  python test/test_speed_det_pose.py [循环次数，默认 100]
"""
import os
import sys
import time

import cv2

from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer

VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "left_side_basketball.mp4")


def pick_player(dets, cls_id=0):
    """取 player 类别中面积最大的框（检测输出坐标系：cpp=640x360 / lite=640x640）。"""
    cand = [d for d in dets if d['cls'] == cls_id]
    if not cand:
        return None
    best = max(cand, key=lambda d: (d['box'][2] - d['box'][0])
               * (d['box'][3] - d['box'][1]))
    return best['box']


def main():
    loops = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    analyzer = VideoAnalyzer()
    analyzer.load_models()
    det = analyzer.det_model
    pose = analyzer.pose_model
    is_cpp = getattr(analyzer, 'det_engine', 'rknn_lite') == 'cpp'
    engine_name = "cpp" if is_cpp else "rknn_lite"

    # 读一帧「有 player」的图（姿态耗时需要有效 ROI；用检测确认帧内有人）
    cap = cv2.VideoCapture(VIDEO)
    f = None
    if cap.isOpened():
        for idx in range(0, 1200, 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            tmp360 = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            if is_cpp:
                dets = det.detect_360(tmp360)
            else:
                tmp_canvas = cv2.copyMakeBorder(tmp360, 140, 140, 0, 0,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
                dets = det.detect_on_canvas(tmp_canvas)
            if pick_player(dets) is not None:
                f = frame
                print(f"选中帧 idx={idx}（检测到 player）")
                break
    cap.release()
    if f is None:
        print("读不到视频帧")
        return

    H, W = f.shape[:2]
    print(f"输入帧: {W}x{H}  检测引擎: {engine_name}  循环 {loops} 次\n")

    # 模拟 RGA 输出（循环外，不计时）
    det360 = cv2.resize(f, (640, 360), interpolation=cv2.INTER_LINEAR)
    preview = cv2.resize(f, (1280, 720), interpolation=cv2.INTER_LINEAR)

    def run_once():
        """单次「检测 -> 抠 ROI -> 姿态」，按引擎分支；返回 (pre_ms, det_ms, pose_ms)。"""
        pre_ms = det_ms = 0.0
        # 1) 检测（cpp 无 Python 补边；lite 补黑边）
        if is_cpp:
            t0 = time.perf_counter()
            dets = det.detect_360(det360)
            det_ms = (time.perf_counter() - t0) * 1000.0
        else:
            t0 = time.perf_counter()
            canvas = cv2.copyMakeBorder(det360, 140, 140, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            pre_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            dets = det.detect_on_canvas(canvas)
            det_ms = (time.perf_counter() - t0) * 1000.0

        # 2) 抠 ROI + 姿态
        t0 = time.perf_counter()
        box = pick_player(dets)
        if box is not None:
            x1, y1, x2, y2 = box
            if is_cpp:
                # cpp 返回 640x360 坐标，直接归一化
                nx1, ny1, nx2, ny2 = x1 / 640.0, y1 / 360.0, x2 / 640.0, y2 / 360.0
            else:
                # lite 返回 640x640 画布坐标，减上下补边(140)后归一化
                nx1, ny1, nx2, ny2 = x1 / 640.0, (y1 - 140) / 360.0, x2 / 640.0, (y2 - 140) / 360.0
            cx1, cy1 = max(0, int(nx1 * 1280)), max(0, int(ny1 * 720))
            cx2, cy2 = min(1279, int(nx2 * 1280)), min(719, int(ny2 * 720))
            if cx1 < cx2 and cy1 < cy2:
                pose.detect_crop(preview[cy1:cy2, cx1:cx2])
        pose_ms = (time.perf_counter() - t0) * 1000.0
        return pre_ms, det_ms, pose_ms

    # 预热（消除首帧模型初始化/内存分配开销）
    for _ in range(3):
        run_once()

    # 正式计时
    pre_total = det_total = pose_total = 0.0
    for _ in range(loops):
        pre_ms, det_ms, pose_ms = run_once()
        pre_total += pre_ms
        det_total += det_ms
        pose_total += pose_ms

    avg_pre = pre_total / loops
    avg_det = det_total / loops
    avg_pose = pose_total / loops
    det_pose = avg_det + avg_pose
    total = avg_pre + det_pose

    print("=" * 56)
    print(f"平均耗时（检测引擎: {engine_name}，{loops} 次取均值）")
    print("=" * 56)
    print(f"预处理(补黑边) : {avg_pre:6.2f} ms" + ("（cpp 无此步）" if is_cpp else ""))
    print(f"检测(detect)   : {avg_det:6.2f} ms")
    print(f"姿态(pose)     : {avg_pose:6.2f} ms")
    print("-" * 56)
    print(f"检测+姿态合计  : {det_pose:6.2f} ms  ← 决定能否每帧采样")
    print(f"端到端总耗时   : {total:6.2f} ms")
    print(f"等效 FPS       : {1000.0 / total:6.1f}")

    # 结论提示（按 25fps 摄像头 = 40ms 帧间隔）
    if det_pose <= 25.0:
        tip = "≤25ms：可 FRAME_STRIDE=1（每帧采样），留足余量"
    elif det_pose <= 35.0:
        tip = "25~35ms：建议 FRAME_STRIDE=2（每 2 帧采样）折中"
    else:
        tip = ">35ms：保持 FRAME_STRIDE=3，靠 anti-flicker 的 miss_max 补偿"
    print(f"\n结论建议：{tip}")

    analyzer.release_models()


if __name__ == "__main__":
    main()
