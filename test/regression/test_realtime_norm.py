# -*- coding: utf-8 -*-
"""
实时 RGA 方案 B 坐标链离线验证（test/test_realtime_norm）
=============================================================
用本地视频模拟实时链路的 RGA 两路输出：
  preview = resize(1920x1080 -> 1280x720)   # 模拟 MPP 显示输出（预览/姿态抠图基准）
  det360  = resize(1920x1080 -> 640x360)    # 模拟 RGA 等比缩放（检测输入）

调用 _extract_frame_metrics_norm(det360, preview)，与离线 _extract_frame_metrics
（letterbox 路径）对比归一化坐标，验证「640x360 -> 补黑边 -> letterbox 反算」映射正确。

用法（RK3588 板端，需已编译 mpp_player / 安装 rknn-toolkit-lite2）：
  python test/regression/test_realtime_norm.py
"""
import sys
from pathlib import Path
# 获取当前脚本所在test文件夹的【父目录】=项目根目录，加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os

import cv2

from vision_algorithm.pipeline.video_analyzer import VideoAnalyzer

VIDEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "test_videos", "left_rtsp.mp4")


def norm(box, w, h):
    """框转归一化坐标（0~1），便于跨分辨率对比。"""
    if not box:
        return None
    return tuple(round(v, 3) for v in (box[0] / w, box[1] / h, box[2] / w, box[3] / h))


def main():
    analyzer = VideoAnalyzer()
    analyzer.load_models()

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"打开视频失败: {VIDEO}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"已打开: {VIDEO}  fps={fps:.1f} total≈{total}")

    frame_idx = 0
    sampled = 0
    while sampled < 30:
        ok, f = cap.read()
        if not ok:
            break
        if frame_idx % 30 != 0:
            frame_idx += 1
            continue
        H, W = f.shape[:2]
        preview = cv2.resize(f, (1280, 720))   # 模拟 MPP 显示输出
        det360 = cv2.resize(f, (640, 360))     # 模拟 RGA 等比缩放

        fd_off = analyzer._extract_frame_metrics(f, frame_idx, ts=frame_idx / fps)
        fd_new = analyzer._extract_frame_metrics_norm(
            det360, preview, frame_idx, ts=frame_idx / fps)

        print(f"\n[帧 {frame_idx}] 原帧 {W}x{H}")
        print(f"  离线: player={norm(fd_off['player_box'], W, H)} "
              f"ball={len(fd_off['ball_boxes'])} "
              f"kpts={'Y' if fd_off['kpts'] is not None else 'N'}")
        print(f"  新  : player={norm(fd_new['player_box'], 1280, 720)} "
              f"ball={len(fd_new['ball_boxes'])} "
              f"kpts={'Y' if fd_new['kpts'] is not None else 'N'}")

        # 范围校验：新路径所有框/关键点必须落在 1280x720 预览坐标内
        box = fd_new['player_box']
        if box:
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720, \
                f"player_box 越界: {box}"
        kpts = fd_new['kpts']
        if kpts is not None:
            assert kpts[:, 0].max() <= 1280 and kpts[:, 1].max() <= 720, \
                "kpts 越界"
        frame_idx += 1
        sampled += 1

    cap.release()
    analyzer.release_models()
    print("\n范围校验通过（RANGE_CHECK_OK）")


if __name__ == "__main__":
    main()
