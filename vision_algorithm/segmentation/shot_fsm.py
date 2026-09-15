# -*- coding: utf-8 -*-
"""
投篮 4 态有限状态机（FSM）+ 环形缓存切段
==========================================
阶段 2 重构：由「6 态串行」改为「4 态串行」。

背景：逐帧观察标准投篮动作（shot_2 等）发现，真实动作存在时间重叠——
    下蹲（下肢）与上举（上肢）并行推进，而非先后串行。6 态串行 + illegal_jump
    会把「下蹲中手腕过顶」误判为跳级，导致完整动作链走不完（shot_count=0）。

状态链（4 态动作链 + IDLE）：
    IDLE -> HOLD -> SQUAT_RAISE -> OVERHEAD_RELEASE -> FOLLOW -> IDLE

    - HOLD              持球：球贴腕、肘屈，尚未启动
    - SQUAT_RAISE       下蹲+上举：下肢屈膝下蹲与上肢上举并行（吸收原 SQUAT/EXTEND/ELBOW_UP）
    - OVERHEAD_RELEASE  过顶+出手：手过头到球离手（吸收原 OVERHEAD/RELEASE）
    - FOLLOW            跟随：球出手后手臂回落（过渡缓冲态，不参与投篮计数判定）

    投篮计数判定：走完 HOLD -> SQUAT_RAISE -> OVERHEAD_RELEASE 三态即判为一次投篮
    （shot_count +1 发生在进入 OVERHEAD_RELEASE 时）。FOLLOW 缺失（如 vis=0 判不出
    出手）不影响 shot_count，仅影响出手帧 release_idx 的精度（退化为手过顶帧）。

关键事件（态内记录，供评分模块用）：
    - hold_idx          持球确认帧
    - crouch_min_idx    下蹲最低点帧（膝角极小值，在 SQUAT_RAISE 内）
    - overhead_idx      手过顶帧（= OVERHEAD_RELEASE 进入帧）
    - release_idx       球离手帧（在 OVERHEAD_RELEASE 内检测球腕分离）

对外接口：
    - ShotFSM(params=None)          # params 为 dict 时直接传入；None 时从 Config.FSM / fsm.yaml 读取
    - feed(frame_dict)              # 喂入一帧，返回逐帧状态 dict（含 shot_event）
    - finalize() / reset()          # 结束/复位

与旧 ShotSegmenter 的契约兼容：
    feed() 的返回值为 dict（非 None），其中 shot_event 字段在检测到一次完整投篮时
    为「投篮段 seg dict」，结构与旧 ShotSegmenter.feed 返回值一致：
        shot_idx / start_idx / release_idx / start_time / end_time / duration /
        frame_metrics / has_squat
    因此下游 _split_and_height / ScoringEngine 无需改动。

依赖：numpy / config（可选）/ vision_algorithm.segmentation.shot_segmenter(KeypointRingBuffer)
"""

import os
from collections import deque

import numpy as np

from vision_algorithm.pose.pose_feature import calculate_angle
from vision_algorithm.segmentation.shot_segmenter import KeypointRingBuffer

try:
    from config import Config
except Exception:  # pragma: no cover - standalone 测试时无 config 包
    Config = None


class ShotFSM:
    """投篮 4 态 FSM：逐帧推进，输出状态与投篮段（shot_event）。"""

    # 状态集合与合法后继（线性链）
    STATES = ['IDLE', 'HOLD', 'SQUAT_RAISE', 'OVERHEAD_RELEASE', 'FOLLOW']
    NEXT_STATE = {
        'IDLE': 'HOLD',
        'HOLD': 'SQUAT_RAISE',
        'SQUAT_RAISE': 'OVERHEAD_RELEASE',
        'OVERHEAD_RELEASE': 'FOLLOW',
        'FOLLOW': 'IDLE',
    }

    # COCO 17 关键点索引
    NOSE = 0
    L_EYE, R_EYE = 1, 2
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    def __init__(self, params=None, ring=None, lookback=None):
        """初始化 4 态 FSM。

        参数：
            params   -- dict，结构与 config/fsm.yaml 的 FSM 节点一致；None 时自动加载
            ring     -- 关键点环形缓存（默认新建 KeypointRingBuffer）
            lookback -- 出手回溯窗口帧数（默认取 Config.LOOKBACK_WINDOW_FRAMES）
        """
        if params is not None:
            self.cfg = params
        else:
            self.cfg = self._load_params()
        self._validate_cfg()

        self.g = self.cfg['GLOBAL']

        self.ring = ring or KeypointRingBuffer()
        self.lookback = lookback
        if self.lookback is None:
            try:
                self.lookback = Config.LOOKBACK_WINDOW_FRAMES
            except Exception:
                self.lookback = 60

        # 运行时状态
        self.state = 'IDLE'
        self.cand_count = 0          # 进入下一态的连续满足帧数（迟滞计数）
        self.hold_frames = 0         # HOLD 累计帧数（超时作废用）
        self.follow_frames = 0       # FOLLOW 累计帧数（超时回待机用）
        self.overhead_frames = 0     # OVERHEAD_RELEASE 累计帧数（超时兜底用）
        self.shot_count = 0          # 已输出的投篮事件数
        self.state_entries = {}      # 各状态最近一次进入的帧号
        self.crouch_min_idx = None   # 下蹲最低点帧号
        self.crouch_min_knee = None  # 下蹲最低点膝角
        self.release_pending_idx = None  # 球腕首次分离帧号
        self.release_count = 0       # 球腕分离连续帧数
        self.lost_count = 0          # 连续「无人且无球」帧计数（检测丢失 watchdog）
        self._last_wrist = None      # 最近有效腕点 (wx, wy)，用于腕点丢失时补位
        self._last_torso_len = None  # 最近有效 torso_len，用于腕点丢失帧的归一化
        self.history = deque(maxlen=max(self.g.get('slope_window', 5), 2))

    # ------------------------------------------------------------------
    # 配置加载与校验
    # ------------------------------------------------------------------
    @staticmethod
    def _load_params():
        """无外部 params 且 Config 不可用时，自动加载 config/fsm.yaml。"""
        import yaml
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))  # 项目根目录
        path = os.path.join(root, 'config', 'fsm.yaml')
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data.get('FSM', {})

    def _validate_cfg(self):
        """校验配置结构，缺关键字段时抛 KeyError。"""
        required = ['STATES', 'TRANSITIONS', 'GLOBAL']
        for key in required:
            if key not in self.cfg:
                raise KeyError(f"FSM 配置缺少关键字段: {key}")

        if self.cfg['STATES'] != self.STATES:
            raise ValueError(
                f"STATES 顺序必须是 {self.STATES}, 实际为 {self.cfg['STATES']}"
            )

        for s in self.STATES:
            legal = self.cfg['TRANSITIONS'].get(s)
            expected = [self.NEXT_STATE[s]]
            if legal != expected:
                raise ValueError(
                    f"TRANSITIONS[{s}] 应为 {expected}, 实际为 {legal}"
                )

        for s in self.STATES[1:]:
            if s not in self.cfg:
                raise KeyError(f"FSM 配置缺少状态参数: {s}")

        if 'INTERRUPTS' not in self.cfg:
            self.cfg['INTERRUPTS'] = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def reset(self):
        """复位到 IDLE，清空历史与计数。"""
        self.state = 'IDLE'
        self.cand_count = 0
        self.hold_frames = 0
        self.follow_frames = 0
        self.state_entries.clear()
        self.crouch_min_idx = None
        self.crouch_min_knee = None
        self.release_pending_idx = None
        self.release_count = 0
        self.lost_count = 0
        self._last_wrist = None
        self._last_torso_len = None
        self.history.clear()

    def finalize(self):
        """录制/视频结束：清空缓存与状态。"""
        self.reset()
        self.ring.clear()

    def feed(self, fd):
        """喂入一帧特征 dict，返回逐帧状态 dict。

        返回 dict：
            {
                'frame_idx': ...,
                'state': ...,                 # 当前 4 态
                'shot_count': ...,
                'features': {...},            # 本帧推导出的归一化特征
                'shot_event': None | {...}    # 完成一次投篮时 = 投篮段 seg dict
            }
        """
        self.ring.push(fd)
        f = self._derive_features(fd)
        frame_idx = fd.get('idx')
        self.history.append({'idx': frame_idx, 'f': f})

        # ── 全局检测丢失 watchdog：连续无人且无球 -> 重置 IDLE ──
        self._update_lost_count(f)

        # ── OVERHEAD_RELEASE 态内：投篮已完成（进入该态即计数），
        #    此处仅精化出手帧 release_idx 并触发切段。vis=0 判不出出手时，
        #    靠超时兜底用「手过顶帧」作为 release_idx，仍能正常切段 + 计数，
        #    保证缺少 FOLLOW 也不影响 shot_count。
        if self.state == 'OVERHEAD_RELEASE':
            self.overhead_frames += 1
            if self._release_detected(f):
                if self.release_count == 0:
                    self.release_pending_idx = frame_idx
                self.release_count += 1
            else:
                self.release_count = 0
                self.release_pending_idx = None

            release_confirmed = (self.release_count >= self.cfg['OVERHEAD_RELEASE'].get(
                'release_confirm_frames', 2))
            timeout = self.overhead_frames > self.g.get('overhead_timeout_frames', 30)
            if release_confirmed or timeout:
                if release_confirmed and self.release_pending_idx is not None:
                    release_idx = self.release_pending_idx
                else:
                    # 超时兜底：用 OVERHEAD_RELEASE 进入帧（手过顶帧）作为出手帧
                    release_idx = self.state_entries.get('OVERHEAD_RELEASE', frame_idx)
                self._transition('FOLLOW', frame_idx)  # 转缓冲态，回 IDLE 用
                seg = self._build_segment(release_idx)
                self.release_count = 0
                self.release_pending_idx = None
                self.overhead_frames = 0
                return self._make_result(frame_idx, f, shot_event=seg)
            return self._make_result(frame_idx, f)

        # ── FOLLOW -> IDLE：手臂回落 / 超时 ──
        if self.state == 'FOLLOW':
            self.follow_frames += 1
            done = self._follow_done(f)
            if done or self.follow_frames > self.g.get('follow_timeout_frames', 40):
                self._reset_to_idle()
            return self._make_result(frame_idx, f)

        # ── HOLD 全局超时：持球后长期未启动 -> 作废 ──
        if self.state == 'HOLD':
            self.hold_frames += 1
            if self.hold_frames > self.g.get('hold_timeout_frames', 90):
                self._reset_to_idle()
                return self._make_result(frame_idx, f)

        # ── 检查进入下一态 ──
        next_state = self.NEXT_STATE[self.state]
        if self._check_entry(next_state, f):
            self.cand_count += 1
            if self.cand_count >= self._entry_hysteresis(next_state):
                self._transition(next_state, frame_idx)
                self.cand_count = 0
        else:
            self.cand_count = 0
            if self._check_interrupt(f):
                self._reset_to_idle()

        # ── SQUAT_RAISE 内跟踪下蹲最低点 ──
        if self.state == 'SQUAT_RAISE':
            self._track_crouch_min(f, frame_idx)

        return self._make_result(frame_idx, f)

    # ------------------------------------------------------------------
    # 特征推导（从原始关键点 + 球框 + 球员框）
    # ------------------------------------------------------------------
    def _derive_features(self, fd):
        """从 frame dict 推导 FSM 需要的归一化特征。"""
        f = {'frame_idx': fd.get('idx')}

        kpts = fd.get('kpts')
        if kpts is None:
            return f
        kpts = np.asarray(kpts, dtype=np.float32)
        if kpts.ndim != 2 or kpts.shape[0] < 17:
            return f

        kpt_conf = fd.get('kpt_conf')
        conf = np.asarray(kpt_conf, dtype=np.float32).reshape(-1) if kpt_conf is not None else None

        n = kpts.shape[0]
        visible = []
        for i in range(n):
            coord_ok = (kpts[i, 0] > 0) or (kpts[i, 1] > 0)
            conf_ok = (conf[i] > 0.0) if (conf is not None and i < len(conf)) else True
            visible.append(bool(coord_ok and conf_ok))
        f['n_visible_kpts'] = int(sum(visible))

        # 篮球检测特征（基础，不依赖关键点/torso_len）：
        # 球检测与姿态检测是两路独立输出。出手瞬间姿态关键点常整帧清空（nk=0），
        # 但球框仍在画面（球向上抛）。若把球框特征放在 torso_len 校验之后，
        # 关键点全丢会连带把 ball_cy 一起丢掉，导致「球框抛射运动」判据失效。
        # 因此先提取球框与置信度，即使 torso_len 无效（后续 return）也保留。
        ball_boxes = fd.get('ball_boxes') or []
        ball_confs = fd.get('ball_confs') or []
        if ball_boxes:
            if len(ball_confs) == len(ball_boxes):
                best_i = int(np.argmax(ball_confs))
                best_conf = float(ball_confs[best_i])
            else:
                best_i = 0
                best_conf = 1.0
            f['ball_conf'] = best_conf
            if best_conf >= self.g.get('ball_conf_min', 0.25):
                bx1, by1, bx2, by2 = ball_boxes[best_i]
                f['ball_box'] = (bx1, by1, bx2, by2)
                f['ball_cx'] = (bx1 + bx2) / 2.0
                f['ball_cy'] = (by1 + by2) / 2.0

        # 投篮臂侧：右腕+右肩可见优先，否则左侧
        if visible[self.R_SHOULDER] and visible[self.R_WRIST]:
            side = 'Right'
            s_i, e_i, w_i, h_i, k_i, a_i = (
                self.R_SHOULDER, self.R_ELBOW, self.R_WRIST,
                self.R_HIP, self.R_KNEE, self.R_ANKLE
            )
        else:
            side = 'Left'
            s_i, e_i, w_i, h_i, k_i, a_i = (
                self.L_SHOULDER, self.L_ELBOW, self.L_WRIST,
                self.L_HIP, self.L_KNEE, self.L_ANKLE
            )
        f['side'] = side

        # torso_len：肩-髋欧氏距离；缺失时回退 player_h
        torso_len = None
        if visible[s_i] and visible[h_i]:
            torso_len = float(np.linalg.norm(kpts[s_i] - kpts[h_i]))
        if torso_len is None or torso_len < self.g.get('torso_min_px', 10.0):
            ph = fd.get('player_h')
            if ph is not None and ph >= self.g.get('torso_min_px', 10.0):
                torso_len = float(ph)
        if torso_len is None or torso_len < self.g.get('torso_min_px', 10.0):
            return f
        f['torso_len'] = torso_len
        self._last_torso_len = float(torso_len)

        # 头部基准 Y：优先双眼均值，其次单眼，最后鼻子
        head_ref_y = None
        eye_ys = []
        if visible[self.L_EYE]:
            eye_ys.append(float(kpts[self.L_EYE, 1]))
        if visible[self.R_EYE]:
            eye_ys.append(float(kpts[self.R_EYE, 1]))
        if eye_ys:
            head_ref_y = sum(eye_ys) / len(eye_ys)
        elif visible[self.NOSE]:
            head_ref_y = float(kpts[self.NOSE, 1])
        f['head_ref_y'] = head_ref_y

        # 关节坐标
        wx, wy = float(kpts[w_i, 0]), float(kpts[w_i, 1])
        ex, ey = float(kpts[e_i, 0]), float(kpts[e_i, 1])
        sx, sy = float(kpts[s_i, 0]), float(kpts[s_i, 1])
        hx, hy = float(kpts[h_i, 0]), float(kpts[h_i, 1])
        kx, ky = float(kpts[k_i, 0]), float(kpts[k_i, 1])
        f['wrist_y'] = wy
        f['wrist_x'] = wx
        if visible[w_i]:
            self._last_wrist = (wx, wy)
        f['shoulder_y'] = sy
        f['hip_y'] = hy
        f['knee_y'] = ky
        f['elbow_y'] = ey

        # 归一化相对高度/距离（÷ torso_len）；正值表示「上方」（y 差取反）
        if head_ref_y is not None and visible[w_i]:
            f['wrist_rel_head'] = (head_ref_y - wy) / torso_len
        if visible[e_i] and visible[s_i]:
            f['elbow_rel_shoulder'] = (sy - ey) / torso_len
        if visible[s_i] and visible[h_i]:
            f['shoulder_rel_hip'] = (hy - sy) / torso_len
        if visible[k_i] and visible[h_i]:
            f['knee_rel_hip'] = (hy - ky) / torso_len

        # 关节角度（任一关键点缺失 -> 该角度 None）
        if visible[s_i] and visible[e_i] and visible[h_i]:
            f['shoulder_ang'] = float(calculate_angle(kpts[h_i], kpts[s_i], kpts[e_i]))
        if visible[s_i] and visible[e_i] and visible[w_i]:
            f['elbow_ang'] = float(calculate_angle(kpts[s_i], kpts[e_i], kpts[w_i]))
        if visible[s_i] and visible[h_i] and visible[k_i]:
            f['hip_ang'] = float(calculate_angle(kpts[s_i], kpts[h_i], kpts[k_i]))
        if visible[h_i] and visible[k_i] and visible[a_i]:
            f['knee_ang'] = float(calculate_angle(kpts[h_i], kpts[k_i], kpts[a_i]))

        # 球腕关系（依赖腕点 + torso_len + 球框基础特征，已在前面提取）
        if visible[w_i] and f.get('ball_cx') is not None:
            bcx = f['ball_cx']
            bcy = f['ball_cy']
            bx1, by1, bx2, by2 = f['ball_box']
            dist = ((bcx - wx) ** 2 + (bcy - wy) ** 2) ** 0.5
            f['ball_wrist_dist_norm'] = dist / torso_len
            intersect = (bx1 <= wx <= bx2) and (by1 <= wy <= by2)
            f['ball_wrist_intersect'] = intersect
            if intersect:
                f['ball_wrist_relation'] = 'intersect'
            elif bcy < wy and f['ball_wrist_dist_norm'] < 0.8:
                f['ball_wrist_relation'] = 'above'
            elif bcy >= wy and f['ball_wrist_dist_norm'] < 0.8:
                f['ball_wrist_relation'] = 'below'
            else:
                f['ball_wrist_relation'] = 'far'

        return f

    # ------------------------------------------------------------------
    # 斜率/速度计算（滑动窗口线性拟合）
    # ------------------------------------------------------------------
    def _slope(self, key):
        if len(self.history) < 2:
            return 0.0
        xs, ys = [], []
        for item in self.history:
            v = item['f'].get(key)
            if v is not None:
                xs.append(float(item['idx']))
                ys.append(float(v))
        if len(xs) < 2:
            return 0.0
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        mx, my = xs.mean(), ys.mean()
        dx = xs - mx
        den = np.dot(dx, dx)
        if den < 1e-6:
            return 0.0
        return float(np.dot(dx, ys - my) / den)

    def _v_down(self, key):
        return self._slope(key)

    def _v_up(self, key):
        return -self._slope(key)

    # ------------------------------------------------------------------
    # 状态转移与结果构造
    # ------------------------------------------------------------------
    def _transition(self, next_state, frame_idx):
        self.state_entries[next_state] = frame_idx
        self.state = next_state
        if next_state == 'HOLD':
            self.hold_frames = 0
        if next_state == 'FOLLOW':
            self.follow_frames = 0
        if next_state == 'SQUAT_RAISE':
            # 进入下蹲段时初始化最低点跟踪
            self.crouch_min_idx = None
            self.crouch_min_knee = None
        if next_state == 'OVERHEAD_RELEASE':
            # 走完 HOLD -> SQUAT_RAISE -> OVERHEAD_RELEASE 三态即判定投篮一次。
            # FOLLOW 仅作回 IDLE 的缓冲，不参与计数判定，故在此提前计数，
            # 避免 vis=0 判不出出手（无 FOLLOW）而丢失该次投篮。
            self.shot_count += 1
            self.overhead_frames = 0

    def _reset_to_idle(self):
        self.state = 'IDLE'
        self.cand_count = 0
        self.hold_frames = 0
        self.follow_frames = 0
        self.state_entries.clear()
        self.crouch_min_idx = None
        self.crouch_min_knee = None
        self.release_pending_idx = None
        self.release_count = 0
        self.lost_count = 0

    def _make_result(self, frame_idx, f, shot_event=None):
        return {
            'frame_idx': frame_idx,
            'state': self.state,
            'shot_count': self.shot_count,
            'features': f,
            'shot_event': shot_event,
        }

    # ------------------------------------------------------------------
    # 各态进入条件
    # ------------------------------------------------------------------
    def _check_entry(self, target, f):
        method = getattr(self, f'_enter_{target.lower()}', None)
        if method is None:
            return False
        return method(f)

    def _entry_hysteresis(self, state):
        """各状态进入迟滞帧数：优先 per-state 的 hysteresis_frames，否则回退 GLOBAL.hysteresis_frames。

        允许对个别状态（如 OVERHEAD_RELEASE）单独放宽/收紧进入迟滞，而不影响
        其他状态的全局默认值，避免全局降迟滞放大关键点抖动带来的误判风险。
        """
        return int(self.cfg.get(state, {}).get(
            'hysteresis_frames', self.g.get('hysteresis_frames', 3)))

    def _enter_hold(self, f):
        """进入 HOLD：球贴腕 + 肘屈（辅助）+ 腕低于肩。"""
        cfg = self.cfg['HOLD']
        if f.get('wrist_y') is None or f.get('ball_conf') is None:
            return False
        if f.get('ball_conf', 0.0) < self.g.get('ball_conf_min', 0.25):
            return False

        near = False
        if f.get('ball_wrist_intersect'):
            near = True
        elif (f.get('ball_wrist_dist_norm') is not None
              and f.get('ball_wrist_dist_norm') < cfg.get('ball_wrist_dist_ratio', 0.30)):
            near = True
        if not near:
            return False

        if cfg.get('require_intersect', True) and not f.get('ball_wrist_intersect'):
            return False

        elbow_ang = f.get('elbow_ang')
        if elbow_ang is not None and elbow_ang > cfg.get('elbow_flex_max', 120.0):
            return False

        if cfg.get('wrist_below_shoulder', True):
            if (f.get('wrist_y') is not None and f.get('shoulder_y') is not None
                    and f.get('wrist_y') < f.get('shoulder_y')):
                return False
        return True

    def _enter_squat_raise(self, f):
        """进入 SQUAT_RAISE：膝角低位/下降，或 腕相对头高度开始上升（上肢启动上举）。"""
        cfg = self.cfg['SQUAT_RAISE']
        knee = f.get('knee_ang')
        if knee is not None and knee < cfg.get('knee_squat_thr', 150.0):
            return True
        if -self._slope('knee_ang') > cfg.get('knee_drop_vel_thr', 2.0):
            return True
        if self._slope('wrist_rel_head') > cfg.get('wrist_rise_vel_thr', 0.03):
            return True
        return False

    def _enter_overhead_release(self, f):
        """进入 OVERHEAD_RELEASE：仅凭手腕高于头（手过顶）。

        球是否离手不在此处判定。初学者 / 真实出手前球可能已开始离手
        （球腕距离已超过「贴手」阈值），若在此处要求球仍贴手（require_ball_near）
        会导致永远无法进入 OVERHEAD_RELEASE，进而卡死在 SQUAT_RAISE。
        「球离手」是 OVERHEAD_RELEASE 态内 _release_detected（出手检测）才关心的事。
        """
        cfg = self.cfg['OVERHEAD_RELEASE']
        wrist_rel = f.get('wrist_rel_head')
        if wrist_rel is None:
            return False
        return wrist_rel > cfg.get('wrist_above_head', 0.0)

    def _enter_follow(self, f):
        """进入 FOLLOW：球腕分离（出手）。正常由出手检测触发，此方法为兼容保留。"""
        return self._release_detected(f)

    # ------------------------------------------------------------------
    # 出手检测
    # ------------------------------------------------------------------
    def _release_detected(self, f):
        """出手判定：多信号 OR 组合，不绑定「球腕距离」单一信号。

        出手瞬间姿态关键点常整帧清空（腕点丢），导致 ball_wrist_dist_norm 缺失，
        单一依赖它会卡死在 OVERHEAD_RELEASE。因此引入多路判据（任一命中即判出手）：
            1. 球框抛射运动（不依赖腕点/肘点）：球中心 cy 快速上升（球向上抛出）。
            2. 球腕距离变化率（腕点可见时）：球快速远离手腕。
            3. 球腕绝对距离（含补旧帧兜底）：球已明显远离；腕点丢时用最近有效腕点补算。
            4. 肘关节伸直（辅助）：肘角快速伸直且接近伸直（甩臂出手）。
        """
        cfg = self.cfg['OVERHEAD_RELEASE']
        if f.get('ball_conf') is None or f.get('ball_conf', 0.0) < self.g.get('ball_conf_min', 0.25):
            return False

        # 判据1：球框抛射运动（不依赖腕点/肘点，最鲁棒）
        if self._ball_rising_fast(cfg):
            return True

        # 判据2：球腕距离变化率（腕点可见时）
        if not f.get('ball_wrist_intersect'):
            sep_vel = self._slope('ball_wrist_dist_norm')
            if sep_vel >= cfg.get('ball_wrist_sep_vel_thr', 0.10):
                return True

            # 判据3：球腕绝对距离兜底（腕点丢时用最近有效腕点补算）
            d = f.get('ball_wrist_dist_norm')
            if d is None:
                d = self._dist_to_last_wrist(f)
            if d is not None and d >= cfg.get('ball_wrist_sep_ratio', 0.80):
                return True

        # 判据4：肘关节伸直辅助（肘点可见时）
        elbow = f.get('elbow_ang')
        if elbow is not None:
            elbow_vel = self._slope('elbow_ang')
            if (elbow_vel >= cfg.get('elbow_extend_vel_thr', 5.0)
                    and elbow >= cfg.get('elbow_extend_ang', 140.0)):
                return True

        return False

    def _ball_rising_fast(self, cfg):
        """球框抛射运动判据：球中心 cy 快速上升（图像 y 减小 = 球向上飞）。

        不依赖腕点/肘点，只在球框可见时用球框自身位置变化判断。出手后球做
        向上抛射，cy 每帧骤降（约 0.25 torso_len/帧）；持球上举时 cy 仅随手腕
        缓慢上升（约 0.08 torso_len/帧）。归一化用最近有效 torso_len（出手帧
        torso_len 常随姿态一起丢，故用 _last_torso_len 缓存）。
        """
        ball_cy_vel = -self._slope('ball_cy')  # 正数 = 球上升
        if ball_cy_vel <= 0:
            return False
        if not self._last_torso_len:
            return False
        norm = ball_cy_vel / self._last_torso_len
        return norm >= cfg.get('ball_rise_vel_thr', 0.15)

    def _dist_to_last_wrist(self, f):
        """腕点丢失时，用最近有效腕点补算球腕距离（补旧帧兜底）。

        只服务于出手判定的距离兜底，不写回 history（避免污染斜率窗口），
        也不改动下游评分的原始关键点。球横向/斜向飞出（cy 变化不占主导）时，
        球 cy 上升判据可能不够，此时球相对最近腕点的欧氏距离仍会快速增大。
        """
        if self._last_wrist is None or self._last_torso_len is None:
            return None
        bcx = f.get('ball_cx')
        bcy = f.get('ball_cy')
        if bcx is None or bcy is None:
            return None
        lwx, lwy = self._last_wrist
        dist = ((bcx - lwx) ** 2 + (bcy - lwy) ** 2) ** 0.5
        return dist / self._last_torso_len

    def _follow_done(self, f):
        """跟随结束：腕回落到头下阈值以下。"""
        cfg = self.cfg['FOLLOW']
        w = f.get('wrist_rel_head')
        return w is not None and w < cfg.get('wrist_drop_below', -0.3)

    def _update_lost_count(self, f):
        """全局检测丢失 watchdog。

        连续「人体关键点全丢 且 篮球检测丢失」超过阈值帧时，强制回 IDLE。
        目的：OVERHEAD_RELEASE / SQUAT_RAISE 无态内超时兜底，若球飞出画面 +
        姿态整帧丢失，会永久卡死；此 watchdog 确保状态机及时复位，避免上次
        卡死状态污染下一次投篮。
        """
        player_vis = f.get('n_visible_kpts', 0) > 0
        ball_vis = (f.get('ball_conf') is not None
                    and f.get('ball_conf', 0.0) >= self.g.get('ball_conf_min', 0.25))
        if not player_vis and not ball_vis:
            self.lost_count += 1
        else:
            self.lost_count = 0

        thr = self.g.get('detect_lost_reset_frames', 150)
        if self.state != 'IDLE' and self.lost_count > thr:
            self._reset_to_idle()

    # ------------------------------------------------------------------
    # 下蹲最低点跟踪
    # ------------------------------------------------------------------
    def _track_crouch_min(self, f, frame_idx):
        knee = f.get('knee_ang')
        if knee is None:
            return
        if self.crouch_min_knee is None or knee < self.crouch_min_knee:
            self.crouch_min_knee = knee
            self.crouch_min_idx = frame_idx

    # ------------------------------------------------------------------
    # 显式中断/回退检测（4 态下几乎不触发，保留结构）
    # ------------------------------------------------------------------
    def _check_interrupt(self, f):
        interrupts = self.cfg.get('INTERRUPTS', {})
        if not interrupts:
            return False
        return False

    # ------------------------------------------------------------------
    # 起点回溯 + 切段（复用旧 ShotSegmenter 的膝角回溯思路）
    # ------------------------------------------------------------------
    @staticmethod
    def _knee_angle(m):
        ang = m.get('angles')
        if not ang or len(ang) < 4:
            return None
        return ang[3]

    def _find_action_start(self, window):
        """反向回溯找「站直 -> 下蹲」分界点 = 动作起点。"""
        if not window:
            return 0
        ordered = sorted(window, key=lambda m: m['idx'])
        n = len(ordered)
        thr = self.cfg['SQUAT_RAISE'].get('knee_squat_thr', 150.0)

        first_low = None
        for i in range(n - 1, -1, -1):
            k = self._knee_angle(ordered[i])
            if k is not None and k < thr:
                first_low = i
                break
        if first_low is None:
            return ordered[0]['idx']

        s = first_low
        while s > 0:
            k_prev = self._knee_angle(ordered[s - 1])
            if k_prev is not None and k_prev < thr:
                s -= 1
            else:
                break
        return ordered[s]['idx']

    def _segment_has_squat(self, metrics):
        thr = self.cfg['SQUAT_RAISE'].get('knee_squat_thr', 150.0)
        for m in metrics:
            k = self._knee_angle(m)
            if k is not None and k < thr:
                return True
        return False

    def _build_segment(self, release_idx):
        """出手确认后回溯切出完整动作段。"""
        window = self.ring.backtrack(self.lookback)
        window_before = [m for m in window if m['idx'] <= release_idx]
        if not window_before:
            window_before = window
        start_idx = self._find_action_start(window_before)
        # 方案1：持球（HOLD）期在动作段中最多保留 hold_keep_frames 帧，更早的持球帧丢弃，
        # 避免持球过久导致动作段（进而小图/骨架图）包含大量冗余静止帧。
        # 同时以 hold_idx 为下限，防止回溯取到持球之前的杂帧（无下蹲时 _find_action_start
        # 会兜底返回窗口首帧，可能早于持球起点）。
        hold_idx = self.state_entries.get('HOLD')
        squat_idx = self.state_entries.get('SQUAT_RAISE')
        if hold_idx is not None and squat_idx is not None:
            keep = int(self.g.get('hold_keep_frames', 5))
            start_idx = max(start_idx, int(hold_idx), int(squat_idx) - keep)
        start_idx = min(start_idx, release_idx)
        metrics = [m for m in window if start_idx <= m['idx'] <= release_idx]
        if not metrics:
            metrics = window_before

        ts_map = {m['idx']: m.get('ts') for m in metrics}
        start_time = ts_map.get(start_idx)
        end_time = ts_map.get(release_idx)
        duration = (end_time - start_time) if (start_time is not None
                                                and end_time is not None) else None
        has_squat = self._segment_has_squat(metrics)
        # shot_count 已在进入 OVERHEAD_RELEASE 时提前 +1（FOLLOW 不参与计数判定），
        # 此处直接复用当前计数作为 shot_idx，不再递增。

        return {
            'shot_idx': self.shot_count,
            'start_idx': start_idx,
            'release_idx': release_idx,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'frame_metrics': metrics,
            'has_squat': has_squat,
            # 4 态分段 + 关键事件（供阶段 3 分析与未来评分细化）
            'state_entries': dict(self.state_entries),
            'hold_idx': self.state_entries.get('HOLD'),
            'crouch_min_idx': self.crouch_min_idx,
            'overhead_idx': self.state_entries.get('OVERHEAD_RELEASE'),
        }
