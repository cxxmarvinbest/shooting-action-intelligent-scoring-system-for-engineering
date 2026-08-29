# -*- coding: utf-8 -*-
"""
视频分析编排层（pipeline）
===========================
职责：串联「描黑边预处理 → 目标检测/跟踪 → 姿态估计 → 特征提取 → 动作分段 → 可视化」，
      产出评分模块所需的 frame_metrics 与角度序列。

对外暴露：VideoAnalyzer

模块划分（本层为编排层，不承载具体算法）：
  - vision_algorithm.preprocess.letterbox.LetterboxPreprocessor  描黑边（1920x1080 -> 544x960）
  - vision_algorithm.detection.det_model.RKNNDetModel            篮球检测（单类别）
  - vision_algorithm.detection.target_selector.TargetSelector    主球员筛选（姿态框）+ 篮球过滤
  - vision_algorithm.pose.pose_model.RKNNPoseModel               人体姿态估计（自带人体框）
  - vision_algorithm.pose.pose_feature                           关节角度 / 左右侧 / 关键特征

依赖：os / cv2 / numpy / config / vision_algorithm.*
"""

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

from config import Config
from common.exceptions import VideoSplitError
from vision_algorithm.common.rknn_infer import letterbox_to_model
from vision_algorithm.mpp.mpp_config import (
    USE_MPP_DECODE, MPP_DISPLAY_W, MPP_DISPLAY_H, MPP_QUEUE_SIZE,
    MPP_SCALE_ENABLED, MPP_SCALE_W, MPP_SCALE_H,
    AUTO_TRANSCODE_H265, FFMPEG_BIN, FFPROBE_BIN)
from vision_algorithm.segmentation.shot_segmenter import ShotSegmenter
from vision_algorithm.detection.det_model import RKNNDetModel
from vision_algorithm.detection.target_selector import TargetSelector
from vision_algorithm.pose.pose_model import RKNNPoseModel
from vision_algorithm.pose.pose_feature import SKELETON_CONNECTIONS, extract_pose_features

logger = logging.getLogger("basketball_scoring")


# 实时识别（RTSP 流硬解 RK3588 mpp）：已启用 _iter_rtsp_frames（见下方方法）。
# mpp_player 的 import 放在 _iter_rtsp_frames / _iter_mpp_frames 函数内懒加载，
# 避免模块级 import mpp_player 导致无 mpp 库的环境（如 Windows 本地）import 失败。


class VideoAnalyzer:
    """视频分析器：检测 + 跟踪 + 姿态 + 特征提取 + 动作分段（RK3588 RKNN 版）"""

    def __init__(self):
        self.det_model = None    # RKNN 目标检测模型
        self.pose_model = None   # RKNN 姿态估计模型
        self.current_fps = 30.0  # 最近一次处理视频的「有效帧率」（= 真实帧率 / 采样步长，供评分用）
        self.video_fps = 30.0    # 最近一次处理视频的「真实帧率」（供 ts 时间戳换算用）
        self.selector = TargetSelector()
        # 最新干净帧（无骨架叠加）+ 对应 AI 识别元数据（供 HTTP /frames 下发）
        self.last_preview_frame = None
        self.last_frame_metrics = None

    def load_models(self):
        """加载 RKNN 检测模型（640x640）与姿态估计模型（320x320）。"""
        logger.info("加载检测模型: %s", Config.DET_RKNN_PATH)
        logger.info("加载姿态模型: %s", Config.POSE_RKNN_PATH)
        core_mask = Config.get("NPU_CORE_MASK", 7)
        logger.info("NPU 核心调度模式: %s", core_mask)
        self.det_model = RKNNDetModel(
            Config.DET_RKNN_PATH,
            conf_thres=Config.DET_CONF_THRES,
            nms_thres=Config.DET_NMS_THRES,
            ball_conf_thres=Config.get("DET_BALL_CONF_THRES", 0.30),
            model_w=640, model_h=640,
            core_mask=core_mask)
        self.pose_model = RKNNPoseModel(
            Config.POSE_RKNN_PATH,
            conf_thres=Config.POSE_CONF_THRES,
            nms_thres=Config.DET_NMS_THRES,
            kpt_conf_thres=Config.get("POSE_KPT_CONF_THRES", 0.5),
            model_w=Config.get("POSE_MODEL_W", 320),
            model_h=Config.get("POSE_MODEL_H", 320),
            core_mask=core_mask)

    def release_models(self):
        """释放两个 RKNN 实例占用的 NPU 资源（进程退出前调用，避免资源泄漏）。"""
        for m in (self.det_model, self.pose_model):
            if m is not None:
                try:
                    m.release()
                except Exception as e:
                    logger.warning("释放 RKNN 模型失败（%s: %s）", type(e).__name__, e)
        self.det_model = None
        self.pose_model = None

    # ============================================================
    # H265 摄像头录制视频 -> 自动转码为 H264 预处理
    # ============================================================
    @staticmethod
    def _probe_video_codec(video_path):
        """用 ffprobe 探测视频编码（返回 'hevc' / 'h264' 等，失败返回 None）。"""
        try:
            out = subprocess.run(
                [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=15)
            return out.stdout.strip().lower() or None
        except Exception:
            return None

    def _transcode_to_h264(self, src_path):
        """把 H265 视频转码为 H264 临时文件，返回新路径；失败返回原路径。"""
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="lq_h264_")
        os.close(fd)
        try:
            os.remove(tmp_path)  # mkstemp 会创建空文件，ffmpeg 需要 -y 覆盖，先删掉更稳
        except Exception:
            pass
        cmd = [FFMPEG_BIN, "-y", "-i", src_path,
               "-c:v", "libx264", "-preset", "fast",
               "-pix_fmt", "yuv420p", tmp_path]
        logger.info("检测到 H265 视频，转码为 H264: %s（源: %s）",
                    os.path.basename(tmp_path), os.path.basename(src_path))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except Exception as e:
            logger.warning("H265 转码异常（%s），回退顺序软解源视频", e)
            return src_path
        if r.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            logger.warning("H265 转码失败，回退顺序软解源视频：%s", r.stderr[-300:])
            return src_path
        return tmp_path

    def _prepare_video(self, video_path):
        """预处理输入视频：若为 H265 且开启自动转码，则转码成 H264 临时文件。

        返回 (work_path, is_temp)；is_temp=True 时调用方负责处理完后删除 work_path。
        """
        if not AUTO_TRANSCODE_H265:
            return video_path, False
        codec = self._probe_video_codec(video_path)
        if codec in ("hevc", "h265"):
            return self._transcode_to_h264(video_path), True
        return video_path, False

    @staticmethod
    def _cleanup_temp(work_path, is_temp):
        """删除转码产生的临时文件（is_temp=True 时）。"""
        if is_temp and work_path:
            try:
                os.remove(work_path)
            except Exception:
                pass

    # ============================================================
    # MPP 硬解（RK3588 硬解 H265 本地文件）
    # ============================================================
    @staticmethod
    def _probe_fps(video_path):
        """用 ffprobe 探测视频平均帧率，失败返回 30.0。"""
        try:
            out = subprocess.run(
                [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=avg_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=15)
            s = out.stdout.strip()
            if "/" in s:
                num, den = s.split("/", 1)
                if float(den) > 0:
                    return float(num) / float(den)
        except Exception:
            pass
        return 30.0

    def _iter_mpp_frames(self, video_path):
        """用 MPP 硬解 H265 本地视频，逐帧 yield (frame_idx, frame)。

        说明：
          - mpp_player.play(url) 走 avformat_open_input，支持本地文件路径；
          - 解码器固定 HEVC，仅适用于 H265 视频；
          - 回调异步，用有界队列在解码线程与主线程间传帧（队列满则阻塞=背压），
            避免「只保留最新帧」丢帧，保证帧号连续、切分准确；
          - frame_id 从 1 开始，这里统一 -1 对齐到 0 起始的 frame_idx。
        """
        import queue
        import sys
        import threading
        import time

        # video_analyzer.py 位于 vision_algorithm/pipeline/，mpp 库在其上级 vision_algorithm/mpp
        mpp_lib_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mpp"))
        if mpp_lib_path not in sys.path:
            sys.path.append(mpp_lib_path)
        import mpp_player

        q = queue.Queue(maxsize=max(1, MPP_QUEUE_SIZE))
        state = {'alive': True, 'error': None}
        stop_evt = threading.Event()  # 停止标志：解除 on_frame 可能阻塞在满队列上的情况

        def on_frame(frame_image, scale_image, frame_id, is_rgb):
            # 队列满则短暂等待（背压），但停止触发后立即退出，不再阻塞解码线程，
            # 避免解码线程卡在满队列导致 MPP 解码队列积压无法释放。
            item = (frame_id, frame_image.copy())
            while not stop_evt.is_set():
                try:
                    q.put(item, timeout=0.2)
                    return
                except queue.Full:
                    continue
            # stop 已触发：丢弃本帧，直接返回

        def on_error(err):
            state['error'] = err
            state['alive'] = False

        player = mpp_player.MppPlayer()
        player.set_callback_frame(on_frame)
        player.set_callback_error(on_error)

        def _play():
            try:
                ok = player.play(video_path,
                                 display_width=MPP_DISPLAY_W,
                                 display_height=MPP_DISPLAY_H,
                                 is_rgb=False)
                if not ok:
                    state['error'] = 'player.play 返回 False（RGA 未初始化或重复播放）'
                    return
                # play 为非阻塞，轮询等待解码线程结束（视频播完 / stop / 出错）
                while player.is_running():
                    time.sleep(0.05)
            except Exception as e:
                state['error'] = str(e)
            finally:
                state['alive'] = False

        threading.Thread(target=_play, daemon=True).start()

        try:
            while True:
                try:
                    frame_id, frame = q.get(timeout=1.0)
                except queue.Empty:
                    if not state['alive']:
                        break
                    continue
                yield frame_id - 1, frame
                if not state['alive'] and q.empty():
                    break
        finally:
            # 1) 先置停止标志，解除解码线程在满队列上的阻塞
            stop_evt.set()
            # 2) 清空残留队列，释放帧引用，避免积压
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            # 3) 停止并关闭播放器（内部清空 MPP 解码队列）
            try:
                player.stop()
            except Exception as e:
                logger.warning("MPP player.stop 异常: %s", e)
            try:
                player.close()
            except Exception as e:
                logger.warning("MPP player.close 异常: %s", e)

    def _process_video_multi_mpp(self, video_path, out_dir, save_visuals):
        """MPP 硬解处理 H265 多投篮视频：第一遍切分，第二遍渲染。"""
        fps = self._probe_fps(video_path)
        stride = max(1, int(Config.FRAME_STRIDE))
        self.current_fps = fps / stride
        self.video_fps = fps

        logger.info("MPP 硬解处理 H265 多投篮视频: %s (fps=%.1f, 采样步长=%d)",
                    os.path.basename(video_path), fps, stride)

        segmenter = ShotSegmenter()
        shots = []

        for frame_idx, frame in self._iter_mpp_frames(video_path):
            if frame_idx % stride != 0:
                continue
            fd = self._extract_frame_metrics(frame, frame_idx, ts=frame_idx / fps)
            seg = segmenter.feed(fd)
            if seg is None:
                continue
            if len(seg['frame_metrics']) < Config.MIN_SHOT_FRAMES:
                logger.warning("丢弃过短段: 起点=%d, 出手=%d, 段长=%d 帧 < %d",
                               seg['start_idx'], seg['release_idx'],
                               len(seg['frame_metrics']), Config.MIN_SHOT_FRAMES)
                continue
            if not seg.get('has_squat'):
                logger.warning("丢弃无真实下蹲的误检段: 起点=%d, 出手=%d, 段长=%d 帧",
                               seg['start_idx'], seg['release_idx'],
                               len(seg['frame_metrics']))
                continue
            seq1, seq2, rel_height, idx_squat = self._split_and_height(
                seg['frame_metrics'])
            seg.update({
                'idx_squat': idx_squat,
                'seq1': np.array(seq1),
                'seq2': np.array(seq2),
                'rel_height': rel_height,
            })
            shots.append(seg)
            logger.info("检测到第 %d 次投篮: 起点=%d, 出手=%d, 分界=%d, 段长=%d 帧",
                        seg['shot_idx'], seg['start_idx'], seg['release_idx'],
                        idx_squat, len(seg['frame_metrics']))

        segmenter.finalize()
        logger.info("MPP 硬解切分完成：共检测到 %d 次投篮", len(shots))

        if not shots:
            return shots

        if save_visuals and out_dir:
            try:
                self._render_segments_mpp(video_path, shots, out_dir)
            except Exception as e:
                logger.warning("MPP 渲染失败（%s: %s），跳过可视化输出", type(e).__name__, e)

        return shots

    def _render_segments_mpp(self, video_path, shots, out_dir):
        """MPP 第二遍顺序播，单遍扫描同时渲染所有投篮段（clip1/clip2 + 帧图）。"""
        fps = self._probe_fps(video_path)
        stride = max(1, int(Config.FRAME_STRIDE))
        slow_fps = (fps / stride) * Config.OUT_SLOW_FACTOR
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writers = {}
        for seg in shots:
            shot_id = seg['shot_idx']
            videos_dir = os.path.join(out_dir, "videos", f"shot{shot_id}")
            frames_dir = os.path.join(out_dir, "frames", f"shot{shot_id}")
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            out1_path = os.path.join(videos_dir, "clip1_squat.mp4")
            out2_path = os.path.join(videos_dir, "clip2_release.mp4")
            out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps,
                                   (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
            out2 = cv2.VideoWriter(out2_path, fourcc, slow_fps,
                                   (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
            writers[shot_id] = (out1, out2, frames_dir)
            seg['clip1_path'] = out1_path
            seg['clip2_path'] = out2_path

        seg_infos = [{
            'shot_id': seg['shot_idx'],
            'start': seg['start_idx'],
            'release': seg['release_idx'],
            'idx_squat': seg['idx_squat'],
            'metric_by_idx': {m['idx']: m for m in seg['frame_metrics']},
        } for seg in shots]

        for frame_idx, frame in self._iter_mpp_frames(video_path):
            if frame_idx % stride != 0:
                continue
            for info in seg_infos:
                if not (info['start'] <= frame_idx <= info['release']):
                    continue
                drawn = frame.copy()
                data = info['metric_by_idx'].get(frame_idx)
                phase_label = 'P1: Squat' if frame_idx <= info['idx_squat'] else 'P2: Release'
                self._draw_annotations(drawn, data, phase_label)
                drawn = cv2.resize(drawn, (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
                out1, out2, frames_dir = writers[info['shot_id']]
                if frame_idx <= info['idx_squat']:
                    out1.write(drawn)
                else:
                    out2.write(drawn)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg"), drawn)

        for out1, out2, _ in writers.values():
            out1.release()
            out2.release()
        logger.info("MPP 可视化输出完成：共渲染 %d 段", len(shots))

    # ============================================================
    # 实时识别：RTSP 流硬解取帧（RK3588 mpp 硬件解码）
    # ------------------------------------------------------------
    # 供实时识别（http_server.py）拉流用。注意：
    #   1) mpp 回调帧默认 is_rgb=False -> BGR，与 cv2.VideoCapture.read() 口径一致，
    #      后续描黑边预处理/推理无需改动。
    #   2) 实时流无法 seek，可视化第二遍重读在流场景下应关闭（save_visuals=False）。
    #   3) 只保留最新一帧：实时推理慢于解码时丢旧帧，避免帧堆积。
    # ============================================================
    def _iter_rtsp_frames(self, rtsp_url, display_w=1920, display_h=1080):
        import sys
        import time
        import threading

        # mpp 库懒加载：避免无 mpp 的环境 import 本模块即失败
        # video_analyzer.py 位于 vision_algorithm/pipeline/，mpp 库在其上级 vision_algorithm/mpp
        mpp_lib_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mpp"))
        if mpp_lib_path not in sys.path:
            sys.path.append(mpp_lib_path)
        import mpp_player

        state = {'frame': None, 'scale': None, 'alive': True}
        lock = threading.Lock()

        def on_frame(frame_image, scale_image, frame_id, is_rgb):
            # 只保留最新一帧（实时评分：推理慢于解码时丢旧帧，防帧堆积）
            with lock:
                state['frame'] = frame_image.copy()
                if (scale_image is not None and scale_image.shape[0] > 0
                        and scale_image.shape[1] > 0):
                    state['scale'] = scale_image.copy()

        def on_error(err):
            state['alive'] = False

        player = mpp_player.MppPlayer()
        player.set_callback_frame(on_frame)
        player.set_callback_error(on_error)

        def _play():
            # play() 是非阻塞的（启动解码子线程后立即返回 true），要轮询等解码线程结束。
            # 若 play() 返回后立刻 state['alive']=False，主线程会立刻退出 generator
            # → finally 里 player.stop() 触发 stop_flag_=true → 正在 RTSP 协商的
            # avformat_open_input 被中断 → "无法打开 RTSP 流"。
            try:
                ok = player.play(
                    rtsp_url, display_width=display_w, display_height=display_h,
                    is_rgb=False,
                    is_mpp_scale_img=MPP_SCALE_ENABLED,
                    mpp_scale_w=MPP_SCALE_W, mpp_scale_h=MPP_SCALE_H)
                if not ok:
                    return
                while player.is_running():
                    time.sleep(0.05)
            except Exception as e:
                logger.warning("MPP RTSP 拉流线程异常: %s", e)
            finally:
                state['alive'] = False

        threading.Thread(target=_play, daemon=True).start()

        # RTSP 流帧率（评分模块 fps 参数用）；如需精确可读 player.get_width/get_height
        self.current_fps = 30.0

        try:
            while state['alive']:
                with lock:
                    frame = state['frame']
                    state['frame'] = None
                    scale = state['scale']
                    state['scale'] = None
                if frame is not None:
                    yield frame, scale
                else:
                    time.sleep(0.005)
        finally:
            # 拉流结束：先标记不再存活并清空残留帧，再停/关播放器，
            # 确保 MPP 解码器内部队列被 stop() 清空释放，避免下次重开队列满。
            with lock:
                state['alive'] = False
                state['frame'] = None
                state['scale'] = None
            try:
                player.stop()
            except Exception as e:
                logger.warning("MPP player.stop 异常: %s", e)
            try:
                player.close()
            except Exception as e:
                logger.warning("MPP player.close 异常: %s", e)


    def process_video(self, video_path, save_visuals=True, out_dir=None):
        """
        处理单个视频：描黑边预处理 + 检测 + 姿态 + 特征提取 + 动作分段（可选可视化）。

        返回（与原版接口一致）：
            (s1, s2, out1_path, out2_path, rel_height, frame_metrics)
        """
        # 1. 读取视频（本地文件，软解）
        #    H265 摄像头录制视频先自动转码为 H264，避免软解打不开/花屏。
        #    RTSP 流硬解：改用上面 _iter_rtsp_frames(rtsp_url) 生成器替代下面
        #    cap 与 while 循环，例如：
        #    for frame in self._iter_rtsp_frames("rtsp://.../av_stream"):
        work_path, is_temp = self._prepare_video(video_path)
        cap = cv2.VideoCapture(work_path)
        if not cap.isOpened():
            self._cleanup_temp(work_path, is_temp)
            return None, None, None, None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(Config.FRAME_STRIDE))
        # 隔帧采样后，相邻采样帧的时间间隔变为 stride/fps 秒；
        # 评分里的时长、角速度都基于帧数×fps 计算，因此把有效帧率设为 fps/stride，保证口径不变。
        self.current_fps = fps / stride
        self.video_fps = fps

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

            frame_metrics.append(
                self._extract_frame_metrics(frame, frame_idx, ts=frame_idx / fps))
            frame_idx += 1

        # 5. 动作分段（分界帧 + seq1/seq2 + 相对出手高度）
        seq1, seq2, rel_height, idx_squat = self._split_and_height(frame_metrics)

        logger.info("动作分段完成: 分界帧=%d, 阶段1=%d 帧, 阶段2=%d 帧",
                    idx_squat, len(seq1), len(seq2))

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

            curr_idx = 0     # 原始帧序号（用于与 idx_squat 比较、命名图片）
            sampled_idx = 0  # frame_metrics 中的采样序号（隔帧后两者不再相等）
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if curr_idx % stride != 0:
                    curr_idx += 1
                    continue

                # 直接用原图叠加标注（frame_metrics 坐标为原图坐标系，二者一致）
                data = frame_metrics[sampled_idx] if sampled_idx < len(frame_metrics) else None
                phase_label = 'P1: Squat' if curr_idx <= idx_squat else 'P2: Release'
                self._draw_annotations(frame, data, phase_label)

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
        self._cleanup_temp(work_path, is_temp)
        s1 = np.array(seq1)
        s2 = np.array(seq2)
        return s1, s2, out1_path, out2_path, rel_height, frame_metrics

    # ============================================================
    # 单帧特征提取 / 分段构造 / 可视化（供单投篮与多投篮复用）
    # ============================================================
    def _extract_frame_metrics(self, frame, frame_idx, frame_orig=None, ts=None):
        """单帧：检测(2类) -> 主球员裁剪姿态 -> 特征，返回 frame_metrics 的一帧 dict。

        坐标链路（三种坐标系不可混淆）：
          - 原图 / 原始视频帧：输入 source，也是最终输出坐标（Qt 显示、上层业务）
          - 640x640 带黑边图：检测模型输入，检测框坐标为此坐标系
          - pose 小图 320x320：姿态模型输入，关键点坐标为此坐标系

        流程（低延迟，串行检测+姿态，检测+姿态合计 ~40ms 内）：
          1. 原图 -> 长边 640、短边等比、补黑边 -> 640x640 画布（免 YOLO 内部再预处理）
          2. 检测（0=player, 1=basketball），框 = 640x640 坐标
          3. 只取 player 框，反算回原图坐标，从【原图】抠 ROI（非 640 画布）
          4. ROI「保持宽高比 + 黑边填充」letterbox 到 320x320 -> pose -> 关键点
          5. 关键点反算：320x320 -> 原图坐标（等价于「原图 -> 640 -> 反算回原图」往返）
          6. 输出 player_box / ball_boxes / kpts 均为原图坐标，preview = 原图

        参数：
            frame      —— 原始视频帧（原图，不定分辨率）。检测/姿态/输出都以它为基准。
            frame_orig —— 保留参数（历史兼容），当前忽略，preview 始终用 frame 原图。
            ts         —— 该帧时间戳（秒）。实时=墙钟 epoch 秒；离线=视频内相对秒。
        """

        # 每帧时间戳：实时由调用方传墙钟 epoch 秒；离线缺省按 帧号/真实帧率 推算
        # 「视频内相对时间」（秒）。两种语义由 shot_segmenter.format_shot_time 自动区分。
        if ts is None:
            fps = getattr(self, 'video_fps', 30.0) or 30.0
            ts = frame_idx / fps if fps > 0 else 0.0

        # 工作帧 = 原图（不做竖幅裁剪）
        work = frame
        width = work.shape[1]

        # 1) 长边 640 + 黑边 -> 640x640 画布（描黑边，等比缩放）
        canvas, scale, dw, dh = letterbox_to_model(
            work, self.det_model.model_w, self.det_model.model_h, pad_color=(0, 0, 0))

        # 2) 目标检测（0=player, 1=basketball），坐标 = 640x640 画布
        dets = self.det_model.detect_on_canvas(canvas)

        # 3) 只取主球员框（640x640 画布坐标）
        player_cls = Config.get("DET_PLAYER_CLS_ID", 0)
        player_box_640 = self.selector.select_main_player_box(dets, player_cls)

        # 640x640 画布坐标 -> 原图坐标（反 letterbox：去黑边 + 反缩放）
        def box_to_orig(b640):
            x1 = (b640[0] - dw) / scale
            y1 = (b640[1] - dh) / scale
            x2 = (b640[2] - dw) / scale
            y2 = (b640[3] - dh) / scale
            return (x1, y1, x2, y2)

        # 4) 姿态估计：从【原图】按 player 框抠 ROI -> 保持宽高比 letterbox 320x320 -> pose
        poses = []
        crop_x1 = crop_y1 = 0
        if player_box_640 is not None:
            fx1, fy1, fx2, fy2 = box_to_orig(player_box_640)
            crop_x1 = max(0, int(round(fx1)))
            crop_y1 = max(0, int(round(fy1)))
            crop_x2 = min(width - 1, int(round(fx2)))
            crop_y2 = min(work.shape[0] - 1, int(round(fy2)))
            crop = work[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size > 0 and crop.shape[0] >= 8 and crop.shape[1] >= 8:
                poses = self.pose_model.detect_crop(crop)

        # player 框输出（原图坐标）
        player_box = None
        if player_box_640 is not None:
            px1, py1, px2, py2 = box_to_orig(player_box_640)
            player_box = (int(round(px1)), int(round(py1)),
                          int(round(px2)), int(round(py2)))

        # 篮球筛选（cls==DET_BALL_CLS_ID，长宽比/尺寸/位置约束）
        dets_orig = [{'box': tuple(int(round(v)) for v in box_to_orig(d['box'])),
                      'cls': d['cls'], 'conf': d['conf']} for d in dets]
        ball_boxes = self.selector.filter_balls(dets_orig, player_box, width)

        # 5) 关键点：pose 裁剪图坐标 -> 原图坐标（加裁剪偏移即可）
        main_kpts = None
        kpt_conf = None
        if poses:
            best = max(poses, key=lambda p: (p['box'][2] - p['box'][0])
                       * (p['box'][3] - p['box'][1]))
            kpts = best['kpts'].copy()          # (17,2) 裁剪图坐标
            kpt_conf = best.get('kpt_conf')
            # 不可见点(0,0)保持 0，只对可见点做坐标映射
            visible = (kpts[:, 0] > 0) | (kpts[:, 1] > 0)
            kpts[visible, 0] += crop_x1
            kpts[visible, 1] += crop_y1
            # 越界保护（原图范围内）
            np.clip(kpts[visible, 0], 0, width - 1, out=kpts[visible, 0])
            np.clip(kpts[visible, 1], 0, work.shape[0] - 1, out=kpts[visible, 1])
            main_kpts = kpts

        current_data = {
            'idx': frame_idx, 'ts': ts, 'hip_y': None, 'angles': None, 'kpts': main_kpts,
            'kpt_conf': kpt_conf, 'ankle_angle': None,
            'cx1': 0, 'cy1': 0, 'player_box': player_box, 'ball_boxes': ball_boxes
        }

        # 区分左右侧并计算关节角度等特征（基于原图坐标，角度/相对值不受缩放影响）
        pose_feat = extract_pose_features(main_kpts, player_box)
        current_data.update(pose_feat)

        # 节流诊断：每 30 帧打印一次球/手腕/持球距离，排查「识别不到投篮」
        self._diag_count = getattr(self, "_diag_count", 0) + 1
        if self._diag_count % 30 == 1:
            wrist_x = current_data.get('wrist_x')
            wrist_y = current_data.get('wrist_y')
            player_h = current_data.get('player_h')
            balls = current_data.get('ball_boxes') or []
            ball_info = []
            for b in balls:
                bx1, by1, bx2, by2 = b
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = None
                if wrist_x is not None and wrist_y is not None and wrist_x > 0:
                    dist = ((bcx - wrist_x) ** 2 + (bcy - wrist_y) ** 2) ** 0.5
                ball_info.append((b, round(dist, 1) if dist is not None else None))
            # 持球判定阈值（与 ShotSegmenter 口径一致），便于判断「球是否贴手腕」
            held_thres = None
            if player_h is not None and player_h > 0:
                held_thres = round(Config.get("BALL_WRIST_DIST_RATIO", 0.35) * player_h, 1)
            logger.info(
                "球检测诊断: 检测=%d, 球框=%d, 手腕=(%s,%s), 球员高=%s, 持球阈值=%s, 球心距手腕=%s",
                len(dets), len(balls), wrist_x, wrist_y, player_h, held_thres, ball_info)

        # 预览 = 原图（Qt 显示摄像头原图大小），元数据坐标与原图一致
        preview = work

        # 保存最新干净帧（无骨架叠加）供 HTTP /frames/raw 下发给 Qt 客户端预览
        self.last_preview_frame = preview
        self.last_frame_metrics = current_data

        return current_data

    @staticmethod
    def _split_and_height(frame_metrics):
        """根据 frame_metrics 计算分界帧、seq1/seq2 与相对出手高度。

        返回 (seq1, seq2, rel_height, idx_squat)。
        异常捕获：frame_metrics 为空 / 结构非法时抛 VideoSplitError（输入数据异常）。
        """
        if frame_metrics is None or len(frame_metrics) == 0:
            raise VideoSplitError(
                "切分输入 frame_metrics 为空（输入数据异常）", kind="video_split")
        n = len(frame_metrics)
        # 默认分界帧取中位采样帧号（仅在无「球越过肩线」时兜底）
        idx_squat = frame_metrics[n // 2]['idx'] if n > 0 else 0

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

        return seq1, seq2, rel_height, idx_squat

    @staticmethod
    def _draw_annotations(frame, data, phase_label):
        """在帧上原地叠加球框/球员框/骨架/角度文本/右下角骨架小窗（坐标为帧坐标系）。"""
        if data is None:
            return
        skeleton_connections = SKELETON_CONNECTIONS
        height, width = frame.shape[:2]

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

        if data['angles'] is not None and len(data['angles']) == 4:
            shoulder, elbow, hip, knee = data['angles']
            texts = [
                f"Phase: {phase_label}",
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

    def _render_segment(self, video_path, seg, out_dir):
        """对单个投篮段重读视频、叠加骨架/检测框，输出 clip1/clip2 视频与逐帧图。

        输出到 out_dir/videos/shot{N}/ 与 out_dir/frames/shot{N}/。
        返回 (clip1_path, clip2_path)。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(Config.FRAME_STRIDE))
        slow_fps = (fps / stride) * Config.OUT_SLOW_FACTOR

        shot_id = seg['shot_idx']
        start_idx = seg['start_idx']
        release_idx = seg['release_idx']
        idx_squat = seg['idx_squat']
        metric_by_idx = {m['idx']: m for m in seg['frame_metrics']}

        videos_dir = os.path.join(out_dir, "videos", f"shot{shot_id}")
        frames_dir = os.path.join(out_dir, "frames", f"shot{shot_id}")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        out1_path = os.path.join(videos_dir, "clip1_squat.mp4")
        out2_path = os.path.join(videos_dir, "clip2_release.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps,
                               (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
        out2 = cv2.VideoWriter(out2_path, fourcc, slow_fps,
                               (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))

        # 顺序重读整段（不 seek）：RTSP 录制 / 损坏的 H265 视频若 seek 到非关键帧，
        # HEVC 解码器会报 "Could not find ref with POC" 导致花屏甚至打不开；
        # 从头顺序读可利用最近关键帧正确解码，只在目标区间内写帧。
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr_idx = 0
        while curr_idx <= release_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_idx < start_idx or curr_idx % stride != 0:
                curr_idx += 1
                continue

            data = metric_by_idx.get(curr_idx)
            phase_label = 'P1: Squat' if curr_idx <= idx_squat else 'P2: Release'
            self._draw_annotations(frame, data, phase_label)

            frame = cv2.resize(frame, (Config.OUT_VIDEO_W, Config.OUT_VIDEO_H))
            if curr_idx <= idx_squat:
                out1.write(frame)
            else:
                out2.write(frame)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{curr_idx:04d}.jpg"), frame)
            curr_idx += 1

        out1.release()
        out2.release()
        cap.release()
        return out1_path, out2_path

    # ============================================================
    # 多投篮视频：关键点环形缓存 + 持球/出手状态机逐投切分
    # ============================================================
    def process_video_multi(self, video_path, out_dir=None, save_visuals=True):
        """处理多投篮完整视频，逐投切分出完整动作段。

        返回 List[ShotSegment]，每个元素含：
            shot_idx / start_idx / release_idx / idx_squat /
            seq1 / seq2 / rel_height / frame_metrics / clip1_path / clip2_path
        """
        # 优先 MPP 硬解 H265（跳过软转码）；MPP 失败/未切出投篮则回退软转码+cv2
        codec = self._probe_video_codec(video_path)
        if codec in ("hevc", "h265") and USE_MPP_DECODE:
            try:
                shots_mpp = self._process_video_multi_mpp(video_path, out_dir, save_visuals)
                if shots_mpp:
                    return shots_mpp
                logger.warning("MPP 硬解未切出投篮，回退软转码+cv2")
            except Exception as e:
                logger.warning("MPP 硬解失败（%s: %s），回退软转码+cv2", type(e).__name__, e)

        # 回退：H265 摄像头录制视频先自动转码为 H264，避免软解打不开/花屏
        work_path, is_temp = self._prepare_video(video_path)
        cap = cv2.VideoCapture(work_path)
        if not cap.isOpened():
            self._cleanup_temp(work_path, is_temp)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(Config.FRAME_STRIDE))
        self.current_fps = fps / stride
        self.video_fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        logger.info("处理多投篮视频: %s (fps=%.1f, 采样步长=%d, 总帧≈%d)",
                    os.path.basename(video_path), fps, stride, total_frames)

        # 关键点环形缓存 + 状态机：持续滚动，检测到一次投篮就切一段。
        # 环形缓存 deque(maxlen) 有界，天然防内存耗尽；本地文件顺序读取不产生帧堆积。
        # 实时 RTSP 流的丢帧策略沿用 _iter_rtsp_frames 的「只保留最新帧」模式。
        segmenter = ShotSegmenter()
        shots = []
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            fd = self._extract_frame_metrics(frame, frame_idx, ts=frame_idx / fps)
            seg = segmenter.feed(fd)
            if seg is not None:
                # 过滤过短误检段（如出手前后仅 2 帧的假投篮）
                if len(seg['frame_metrics']) < Config.MIN_SHOT_FRAMES:
                    logger.warning("丢弃过短段: 起点=%d, 出手=%d, 段长=%d 帧 < %d",
                                   seg['start_idx'], seg['release_idx'],
                                   len(seg['frame_metrics']), Config.MIN_SHOT_FRAMES)
                    continue
                if not seg.get('has_squat'):
                    logger.warning("丢弃无真实下蹲的误检段: 起点=%d, 出手=%d, 段长=%d 帧",
                                   seg['start_idx'], seg['release_idx'],
                                   len(seg['frame_metrics']))
                    continue
                seq1, seq2, rel_height, idx_squat = self._split_and_height(
                    seg['frame_metrics'])
                seg.update({
                    'idx_squat': idx_squat,
                    'seq1': np.array(seq1),
                    'seq2': np.array(seq2),
                    'rel_height': rel_height,
                })
                shots.append(seg)
                logger.info("检测到第 %d 次投篮: 起点=%d, 出手=%d, 分界=%d, 段长=%d 帧",
                            seg['shot_idx'], seg['start_idx'], seg['release_idx'],
                            idx_squat, len(seg['frame_metrics']))
            frame_idx += 1

        segmenter.finalize()
        cap.release()
        logger.info("多投篮切分完成：共检测到 %d 次投篮", len(shots))

        if save_visuals and out_dir and shots:
            for seg in shots:
                c1, c2 = self._render_segment(work_path, seg, out_dir)
                seg['clip1_path'] = c1
                seg['clip2_path'] = c2

        self._cleanup_temp(work_path, is_temp)
        return shots
