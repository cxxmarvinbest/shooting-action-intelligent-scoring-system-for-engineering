# -*- coding: utf-8 -*-
"""
视频分析编排层（pipeline）
===========================
职责：串联「描黑边预处理 → 目标检测/跟踪 → 姿态估计 → 特征提取 → 动作分段 → 可视化」，
      产出评分模块所需的 frame_metrics 与角度序列。

对外暴露：VideoAnalyzer

模块划分（本层为编排层，不承载具体算法）：
  - tracker.preprocess.LetterboxPreprocessor  描黑边（1920x1080 -> 544x960）
  - tracker.det_model.RKNNDetModel            球员/篮球检测
  - tracker.target_selector.TargetSelector    主球员筛选 + 篮球过滤
  - pose_estimate.pose_model.RKNNPoseModel    人体姿态估计
  - pose_estimate.pose_feature                关节角度 / 左右侧 / 关键特征

依赖：os / cv2 / numpy / config / tracker.* / pose_estimate.*
"""

import logging
import os
import cv2
import numpy as np

from config import Config
from tracker.preprocess import LetterboxPreprocessor
from tracker.det_model import RKNNDetModel
from tracker.target_selector import TargetSelector
from pose_estimate.pose_model import RKNNPoseModel
from pose_estimate.pose_feature import SKELETON_CONNECTIONS, extract_pose_features

logger = logging.getLogger("basketball_scoring")


# RTSP 流硬解（RK3588 mpp）接入时取消下面注释：
# import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# mpp_lib_path = os.path.abspath(os.path.join(script_dir, "../mpp"))
# sys.path.append(mpp_lib_path)
# import time
# import threading
# import mpp_player


class VideoAnalyzer:
    """视频分析器：检测 + 跟踪 + 姿态 + 特征提取 + 动作分段（RK3588 RKNN 版）"""

    def __init__(self):
        self.det_model = None    # RKNN 目标检测模型
        self.pose_model = None   # RKNN 姿态估计模型
        self.current_fps = 30.0  # 最近一次处理的视频帧率
        self.preprocessor = LetterboxPreprocessor()
        self.selector = TargetSelector()

    def load_models(self):
        """加载 RKNN 检测模型与姿态估计模型"""
        logger.info("加载检测模型: %s", Config.DET_RKNN_PATH)
        logger.info("加载姿态模型: %s", Config.POSE_RKNN_PATH)
        self.det_model = RKNNDetModel(
            Config.DET_RKNN_PATH,
            conf_thres=Config.DET_CONF_THRES,
            nms_thres=Config.DET_NMS_THRES)
        self.pose_model = RKNNPoseModel(
            Config.POSE_RKNN_PATH,
            conf_thres=Config.POSE_CONF_THRES,
            nms_thres=Config.DET_NMS_THRES)

    # ============================================================
    # 【备用】RTSP 流硬解取帧（RK3588 mpp 硬件解码）
    # ------------------------------------------------------------
    # 测试 RTSP 流评分时启用：取消本方法与顶部 import 的注释，再把
    # process_video 里的读本地文件循环替换为
    #     for frame in self._iter_rtsp_frames(rtsp_url):
    # 注意：
    #   1) mpp 回调帧默认 is_rgb=False -> BGR，与 cv2.VideoCapture.read() 口径一致，
    #      后续描黑边预处理/推理无需改动。
    #   2) 实时流无法 seek，可视化第二遍重读在流场景下应关闭（save_visuals=False）。
    # ============================================================
    # def _iter_rtsp_frames(self, rtsp_url, display_w=1920, display_h=1080):
    #     import time
    #     import threading
    #     import mpp_player
    #
    #     state = {'frame': None, 'alive': True}
    #     lock = threading.Lock()
    #
    #     def on_frame(frame_image, scale_image, frame_id, is_rgb):
    #         # 只保留最新一帧（实时评分：推理慢于解码时丢旧帧）
    #         with lock:
    #             state['frame'] = frame_image.copy()
    #
    #     def on_error(err):
    #         state['alive'] = False
    #
    #     player = mpp_player.MppPlayer()
    #     player.set_callback_frame(on_frame)
    #     player.set_callback_error(on_error)
    #
    #     def _play():
    #         # play() 为阻塞式，须在后台线程运行
    #         player.play(rtsp_url, display_width=display_w,
    #                     display_height=display_h, is_rgb=False)
    #         state['alive'] = False
    #
    #     threading.Thread(target=_play, daemon=True).start()
    #
    #     # RTSP 流帧率（评分模块 fps 参数用）；如需精确可读 player.get_width/get_height
    #     self.current_fps = 30.0
    #
    #     try:
    #         while state['alive']:
    #             with lock:
    #                 frame = state['frame']
    #                 state['frame'] = None
    #             if frame is not None:
    #                 yield frame
    #             else:
    #                 time.sleep(0.005)
    #     finally:
    #         player.stop()
    #         player.close()

    def process_video(self, video_path, save_visuals=True, out_dir=None):
        """
        处理单个视频：描黑边预处理 + 检测 + 姿态 + 特征提取 + 动作分段（可选可视化）。

        返回（与原版接口一致）：
            (s1, s2, out1_path, out2_path, rel_height, frame_metrics)
        """
        # 1. 读取视频（本地文件，软解）
        #    RTSP 流硬解：改用上面 _iter_rtsp_frames(rtsp_url) 生成器替代下面
        #    cap 与 while 循环，例如：
        #    for frame in self._iter_rtsp_frames("rtsp://.../av_stream"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None, None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(Config.FRAME_STRIDE))
        # 隔帧采样后，相邻采样帧的时间间隔变为 stride/fps 秒；
        # 评分里的时长、角速度都基于帧数×fps 计算，因此把有效帧率设为 fps/stride，保证口径不变。
        self.current_fps = fps / stride

        logger.info("处理视频: %s (fps=%.1f, 采样步长=%d)",
                    os.path.basename(video_path), fps, stride)

        frame_metrics = []
        frame_idx = 0

        # 2. 逐帧：描黑边预处理 -> 检测 -> 姿态 -> 特征
        #    隔帧采样：仅对 stride 的整数倍帧做检测/姿态/特征分析，其余帧直接跳过。
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            # 横屏摄像头视频 -> 中央竖幅裁剪 -> 544x960（标准视频原样通过）
            frame, _ = self.preprocessor.process(frame)
            width, height = Config.TARGET_W, Config.TARGET_H

            dets = self.det_model.detect(frame)
            poses = self.pose_model.detect(frame)

            current_data = {
                'idx': frame_idx, 'hip_y': None, 'angles': None, 'kpts': None,
                'cx1': 0, 'cy1': 0, 'player_box': None, 'ball_boxes': []
            }

            # 3. 筛选主球员（最大面积人体框）
            player_box, main_kpts = self.selector.select_main_player(poses)
            current_data['player_box'] = player_box
            current_data['kpts'] = main_kpts

            # 篮球筛选（cls==1，长宽比/尺寸/位置约束）
            current_data['ball_boxes'] = self.selector.filter_balls(
                dets, player_box, width)

            # 4. 区分左右侧并计算关节角度等特征
            pose_feat = extract_pose_features(main_kpts, player_box)
            current_data.update(pose_feat)

            frame_metrics.append(current_data)
            frame_idx += 1

        # 5. 判断中间阶段分界帧（球底边越过肩部线）
        idx_squat = frame_idx // 2

        for m in frame_metrics:
            if m.get('shoulder_y') is not None and len(m.get('ball_boxes', [])) > 0:
                bx1, by1, bx2, by2 = m['ball_boxes'][0]
                shoulder_y = m['shoulder_y']
                if by2 < shoulder_y:
                    idx_squat = m['idx']
                    break

        seq1, seq2 = [], []
        for m in frame_metrics:
            if m['angles'] is not None:
                if m['idx'] <= idx_squat:
                    seq1.append(m['angles'])
                else:
                    seq2.append(m['angles'])

        if not seq1:
            seq1 = [[180.0, 180.0, 180.0, 180.0], [180.0, 180.0, 180.0, 180.0]]
        if not seq2:
            seq2 = [[180.0, 180.0, 180.0, 180.0], [180.0, 180.0, 180.0, 180.0]]

        logger.info("动作分段完成: 分界帧=%d, 阶段1=%d 帧, 阶段2=%d 帧",
                    idx_squat, len(seq1), len(seq2))

        # 注意：'wrist_y' in m 只判断 key 是否存在，但 extract_pose_features
        # 在未检测到人时仍会写入 key（值为 None），必须用 is not None 过滤，
        # 否则 min() 会把 None 喂进去导致 '<'/'>' not supported ... NoneType 报错
        valid_height_frames = [m for m in frame_metrics
                               if m.get('wrist_y') is not None
                               and m.get('player_h') is not None]
        if valid_height_frames:
            start_wrist_y = valid_height_frames[0]['wrist_y']
            min_wrist_y = min(m['wrist_y'] for m in valid_height_frames)
            player_h = valid_height_frames[0]['player_h']
            rel_height = (start_wrist_y - min_wrist_y) / player_h if player_h > 0 else 0.0
        else:
            rel_height = 0.0

        out1_path, out2_path = None, None

        if save_visuals and out_dir:
            # 分段视频与逐帧图片分目录存放，避免混在一起
            videos_dir = os.path.join(out_dir, "videos")
            frames_dir = os.path.join(out_dir, "frames")
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)

            # 清理旧的视频/帧图产物（只清理各自子目录，不影响报告/日志/JSON 等结果文件）
            for d in (videos_dir, frames_dir):
                for file_name in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, file_name))
                    except Exception:
                        pass

            # 6. 可视化输出（重新读一遍视频，叠加骨架/检测框后按分段写出）
            #    同样隔帧写出；视频帧率按有效帧率缩放，保持慢放比例不变。
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            slow_fps = (fps / stride) * Config.OUT_SLOW_FACTOR
            out1_path = os.path.join(videos_dir, "clip1_squat.mp4")
            out2_path = os.path.join(videos_dir, "clip2_release.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps,
                                   (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
            out2 = cv2.VideoWriter(out2_path, fourcc, slow_fps,
                                   (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))

            skeleton_connections = SKELETON_CONNECTIONS
            width, height = Config.TARGET_W, Config.TARGET_H

            curr_idx = 0     # 原始帧序号（用于与 idx_squat 比较、命名图片）
            sampled_idx = 0  # frame_metrics 中的采样序号（隔帧后两者不再相等）
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if curr_idx % stride != 0:
                    curr_idx += 1
                    continue

                # 与推理阶段保持完全一致的预处理，保证坐标对得上
                frame, _ = self.preprocessor.process(frame)

                data = frame_metrics[sampled_idx] if sampled_idx < len(frame_metrics) else None
                if data is not None:
                    for bx in data.get('ball_boxes', []):
                        bx1, by1, bx2, by2 = bx
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                        cv2.putText(frame, "Ball", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    if data['player_box'] is not None:
                        px1, py1, px2, py2 = data['player_box']
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 144, 30), 2)
                        cv2.putText(frame, "Player", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 144, 30), 2)

                    if data['kpts'] is not None:
                        for p1_idx, p2_idx in skeleton_connections:
                            pt1, pt2 = data['kpts'][p1_idx], data['kpts'][p2_idx]
                            if pt1[0] == 0 or pt2[0] == 0:
                                continue
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])),
                                     (int(pt2[0]), int(pt2[1])), (220, 110, 0), 2)
                        for i, pt in enumerate(data['kpts']):
                            if pt[0] == 0:
                                continue
                            color, r = ((0, 255, 255), 3) if i <= 4 else ((50, 255, 50), 5)
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), r, color, -1)

                    if data['angles'] is not None:
                        shoulder, elbow, hip, knee = data['angles']
                        texts = [
                            f"Phase: {'P1: Squat' if curr_idx <= idx_squat else 'P2: Release'}",
                            f"Side: {data.get('side_str', 'Unknown')}", f"Shoulder: {shoulder:.1f}",
                            f"Elbow: {elbow:.1f}", f"Hip: {hip:.1f}", f"Knee: {knee:.1f}"
                        ]
                        for i, txt in enumerate(texts):
                            cv2.putText(frame, txt, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                    # 右下角骨架小窗
                    if data['kpts'] is not None:
                        sm_w, sm_h = int(width * 0.20), int(height * 0.35)
                        sm_x1, sm_y1 = width - sm_w - 20, height - sm_h - 20

                        overlay = frame.copy()
                        cv2.rectangle(overlay, (sm_x1, sm_y1), (sm_x1 + sm_w, sm_y1 + sm_h), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        cv2.putText(frame, "Pose", (sm_x1 + 10, sm_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (200, 200, 200), 2)

                        valid_pts = [pt for pt in data['kpts'] if pt[0] > 0]
                        if len(valid_pts) > 0:
                            min_x = min(pt[0] for pt in valid_pts)
                            max_x = max(pt[0] for pt in valid_pts)
                            min_y = min(pt[1] for pt in valid_pts)
                            max_y = max(pt[1] for pt in valid_pts)

                            skel_w = max_x - min_x + 1e-5
                            skel_h = max_y - min_y + 1e-5

                            padding = 25
                            scale = min((sm_w - 2 * padding) / skel_w, (sm_h - 2 * padding) / skel_h)

                            def get_sm_pt(pt):
                                if pt[0] == 0:
                                    return None
                                nx = sm_x1 + sm_w / 2 + (pt[0] - (min_x + skel_w / 2)) * scale
                                ny = sm_y1 + sm_h / 2 + (pt[1] - (min_y + skel_h / 2)) * scale
                                return (int(nx), int(ny))

                            for p1_idx, p2_idx in skeleton_connections:
                                sm_pt1 = get_sm_pt(data['kpts'][p1_idx])
                                sm_pt2 = get_sm_pt(data['kpts'][p2_idx])
                                if sm_pt1 and sm_pt2:
                                    cv2.line(frame, sm_pt1, sm_pt2, (255, 255, 255), 2)

                            for pt in data['kpts']:
                                sm_pt = get_sm_pt(pt)
                                if sm_pt:
                                    cv2.circle(frame, sm_pt, 3, (0, 255, 255), -1)

                frame = cv2.resize(frame, (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
                if curr_idx <= idx_squat:
                    out1.write(frame)
                else:
                    out2.write(frame)

                cv2.imwrite(os.path.join(frames_dir, f"frame_{curr_idx:04d}.jpg"), frame)
                curr_idx += 1
                sampled_idx += 1

            out1.release()
            out2.release()
            logger.info("可视化输出完成: 视频 -> %s / %s；帧图 -> %s 目录",
                        out1_path, out2_path, frames_dir)

        cap.release()
        s1 = np.array(seq1)
        s2 = np.array(seq2)
        return s1, s2, out1_path, out2_path, rel_height, frame_metrics
