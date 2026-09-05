# -*- coding: utf-8 -*-
"""
存储目录布局（common/save_data_layout）
=========================================
负责「save_data/{date}/{session}/{videos|images}/...」分层目录的创建、命名
与路径拼接，是 N1（存储目录完整重构）的核心布局工具。

目录结构规范（与「任务重构整理」长文一致）：
  save_data/
    YYYY-MM-DD/                  # 一级：当天日期文件夹（按本地 %Y-%m-%d 自动创建/复用）
      YYYYMMDD_HHMMSS_userID/    # 二级：单次运动会话
        videos/
          01-YYYYMMDD_HHMMSS-HHMMSS_raw.mp4     # 原始视频
          01-YYYYMMDD_HHMMSS-HHMMSS_ai.mp4      # AI 渲染视频
        images/
          001/                                  # 第 1 次投篮
            58-src.jpg
            58-ai.jpg
            data.json
          002/ ...
          003/ ...

边界条件：
  - 日期目录不存在自动创建（get_today_dir），已存在直接复用
  - 会话目录生成时检查命名冲突（同名则追加 _N 后缀）
  - 投篮序号 / 视频分片序号 自增为三位十进制（001、002 ...）
  - 路径操作只走 os.path / datetime，不引入第三方依赖
  - 任何 mkdir 异常必须抛出，让上层明确处理（不要静默吞）
"""

import datetime
import os
import re
import time
from typing import Optional


# ── 命名常量（与长文严格对齐） ────────────────────────────────────────────
DATE_DIR_FMT = "%Y-%m-%d"                # 一级目录格式：2026-09-03
SESSION_NAME_FMT = "%Y%m%d_%H%M%S"       # 会话目录名时间部分：20260903_150343
VIDEO_NAME_DT_FMT = "%Y%m%d_%H%M%S"      # 视频文件名时间部分：20260903_150343
DEFAULT_USER_ID = "0000"                 # 无 user_id 时的默认值

# 投篮子目录编号：三位十进制，从 1 开始
SHOT_IDX_WIDTH = 3

# 视频分片编号：两位十进制，从 1 开始
CLIP_IDX_WIDTH = 2

# 视频命名：01-20260903_150343-150843_raw.mp4
# raw: 原始视频；ai: AI 渲染视频（带 player 检测框 + 17 点 COCO 骨架）
VIDEO_KIND_RAW = "raw"
VIDEO_KIND_AI = "ai"

# 投篮帧命名：58-src.jpg / 58-ai.jpg
FRAME_KIND_SRC = "src"
FRAME_KIND_AI = "ai"

# 合法日期目录名（用于 N4 磁盘清理脚本识别可清理目录）
DATE_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 合法 user_id：字母数字下划线，长度 1~32（防御性约束，避免路径注入）
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class SaveDataLayoutError(Exception):
    """save_data 路径布局相关错误。"""
    pass


def _sanitize_user_id(user_id: Optional[str]) -> str:
    """清洗 user_id：None / 非法值 → DEFAULT_USER_ID；合法原样返回。

    上层 HTTP 接口允许 user_id 可选；这里做防御性兜底，避免空串、特殊字符
    污染路径或与未来 N4 清理脚本误匹配。
    """
    if not user_id:
        return DEFAULT_USER_ID
    s = str(user_id).strip()
    if not USER_ID_PATTERN.match(s):
        return DEFAULT_USER_ID
    return s


class SaveDataLayout:
    """save_data 目录布局管理。

    用法（与 controller / inference / recording 解耦，仅负责路径与编号）：
        layout = SaveDataLayout(root=Config.SAVE_DATA_ROOT)
        date_dir = layout.get_today_dir()                          # 一级：今天日期
        session_dir, session_name = layout.new_session_dir("1001") # 二级：会话目录
        videos_dir = layout.videos_dir(session_dir)
        images_dir = layout.images_dir(session_dir)
        shot_dir = layout.shot_dir(images_dir, shot_idx=1)
        src_path = layout.shot_frame_path(shot_dir, frame_idx=58, kind="src")
    """

    # ── 类属性常量（供 controller 以 SaveDataLayout.XXX 形式引用）──
    # 与模块层同名常量保持同一份值，仅作为类属性别名暴露。
    # 此前 recording_manage / inference_manage 误以类属性方式访问，
    # 而常量只定义在模块层，导致 AttributeError 使视频/图片全部落盘失败。
    VIDEO_KIND_RAW = VIDEO_KIND_RAW
    VIDEO_KIND_AI = VIDEO_KIND_AI
    FRAME_KIND_SRC = FRAME_KIND_SRC
    FRAME_KIND_AI = FRAME_KIND_AI

    def __init__(self, root: str):
        if not root:
            raise SaveDataLayoutError("save_data 根路径不能为空")
        self.root = os.path.abspath(root)

    # ------------------------------------------------------------------
    # 一级：当天日期目录
    # ------------------------------------------------------------------
    def get_today_dir(self, now: Optional[datetime.datetime] = None) -> str:
        """获取「当天日期」目录，不存在自动创建。已存在直接复用。

        now：可选参数，便于测试注入固定时间；不传则取本地时间。
        返回：绝对路径，例如 D:/.../save_data/2026-09-04
        """
        if now is None:
            now = datetime.datetime.now()
        date_str = now.strftime(DATE_DIR_FMT)
        path = os.path.join(self.root, date_str)
        os.makedirs(path, exist_ok=True)
        return path

    def get_date_dir(self, date_str: str) -> str:
        """获取指定日期目录（不存在自动创建），用于按日期回溯/恢复。

        date_str 形如 '2026-09-04'，与 DATE_DIR_FMT 一致。
        """
        try:
            datetime.datetime.strptime(date_str, DATE_DIR_FMT)
        except ValueError as e:
            raise SaveDataLayoutError(
                f"日期格式非法: {date_str!r}（要求 {DATE_DIR_FMT}）") from e
        path = os.path.join(self.root, date_str)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # 二级：会话目录
    # ------------------------------------------------------------------
    def new_session_dir(self, user_id: Optional[str] = None,
                        now: Optional[datetime.datetime] = None) -> tuple:
        """在当天日期目录下创建「单次运动会话」目录。

        命名规则：YYYYMMDD_HHMMSS_userID
        同名冲突：自动追加 _N 后缀（N 从 2 开始）。

        返回 (session_dir_abs, session_name)，session_name 仅含目录名部分
        （不含日期前缀），供日志 / 状态接口使用。
        """
        if now is None:
            now = datetime.datetime.now()
        date_dir = self.get_today_dir(now=now)
        ts = now.strftime(SESSION_NAME_FMT)
        uid = _sanitize_user_id(user_id)
        base = f"{ts}_{uid}"
        name = base
        suffix = 2
        while os.path.exists(os.path.join(date_dir, name)):
            name = f"{base}_{suffix}"
            suffix += 1
        session_dir = os.path.join(date_dir, name)
        os.makedirs(session_dir, exist_ok=True)
        # 子目录一次性建好
        os.makedirs(os.path.join(session_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
        return session_dir, name

    # ------------------------------------------------------------------
    # 二级目录内的子目录 / 路径拼接
    # ------------------------------------------------------------------
    @staticmethod
    def videos_dir(session_dir: str) -> str:
        """会话内的 videos/ 目录。"""
        return os.path.join(session_dir, "videos")

    @staticmethod
    def images_dir(session_dir: str) -> str:
        """会话内的 images/ 目录。"""
        return os.path.join(session_dir, "images")

    @staticmethod
    def shot_dir(images_dir: str, shot_idx: int) -> str:
        """投篮子目录 images_dir/00N/，存在则复用。

        shot_idx 从 1 开始；若传入 0 或负数，抛 SaveDataLayoutError。
        """
        if shot_idx < 1:
            raise SaveDataLayoutError(
                f"shot_idx 必须 >= 1（实际 {shot_idx}）")
        name = f"{shot_idx:0{SHOT_IDX_WIDTH}d}"
        path = os.path.join(images_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def shot_frame_path(shot_dir: str, frame_idx: int, kind: str) -> str:
        """单帧图片路径：{shot_dir}/{frame_idx:03d}-{kind}.jpg

        kind 仅允许 'src' / 'ai'，其它抛 SaveDataLayoutError。
        frame_idx 不强制 >= 1（允许 0 作占位），但必须是非负整数。
        """
        if kind not in (FRAME_KIND_SRC, FRAME_KIND_AI):
            raise SaveDataLayoutError(
                f"frame kind 非法: {kind!r}（仅 {FRAME_KIND_SRC}/{FRAME_KIND_AI}）")
        if frame_idx < 0:
            raise SaveDataLayoutError(
                f"frame_idx 必须非负（实际 {frame_idx}）")
        return os.path.join(shot_dir, f"{frame_idx:03d}-{kind}.jpg")

    @staticmethod
    def shot_data_json_path(shot_dir: str) -> str:
        """投篮 data.json 路径：{shot_dir}/data.json"""
        return os.path.join(shot_dir, "data.json")

    # ------------------------------------------------------------------
    # 自增编号
    # ------------------------------------------------------------------
    @staticmethod
    def next_shot_idx(images_dir: str) -> int:
        """计算 images_dir/ 下「下一个投篮编号」。

        扫描现有 NNN/ 目录（与 SHOT_IDX_WIDTH 位数一致），取最大 +1；空目录返回 1。
        这样确保多个会话/重启后编号不冲突（基于实际目录推断，非内存计数）。
        """
        if not os.path.isdir(images_dir):
            return 1
        max_idx = 0
        pattern = re.compile(r"^(\d{%d})$" % SHOT_IDX_WIDTH)
        for name in os.listdir(images_dir):
            m = pattern.match(name)
            if not m:
                continue
            sub = os.path.join(images_dir, name)
            if os.path.isdir(sub):
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1

    @staticmethod
    def next_clip_idx(videos_dir: str) -> int:
        """计算 videos_dir/ 下「下一个视频分片编号」。

        扫描 NN-*.mp4 形式的文件名，取最大 +1；空目录返回 1。
        用于实时视频每 5 分钟 rotate 时自动累加。
        """
        if not os.path.isdir(videos_dir):
            return 1
        max_idx = 0
        pattern = re.compile(r"^(\d{%d})-.*\.mp4$" % CLIP_IDX_WIDTH)
        for name in os.listdir(videos_dir):
            m = pattern.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
        return max_idx + 1

    # ------------------------------------------------------------------
    # 视频文件名拼接
    # ------------------------------------------------------------------
    @staticmethod
    def build_video_filename(clip_idx: int, start_ts: float, end_ts: float,
                             kind: str, ext: str = "mp4") -> str:
        """生成视频文件名：01-20260903_150343-150843_raw.mp4

        clip_idx：分片编号（1 起，CLIP_IDX_WIDTH 位）
        start_ts / end_ts：分片起止 epoch 秒（用于格式化时间戳）
        kind: 'raw' / 'ai'
        ext: 视频扩展名（默认 mp4）
        """
        if kind not in (VIDEO_KIND_RAW, VIDEO_KIND_AI):
            raise SaveDataLayoutError(
                f"video kind 非法: {kind!r}（仅 {VIDEO_KIND_RAW}/{VIDEO_KIND_AI}）")
        try:
            s = time.strftime(VIDEO_NAME_DT_FMT, time.localtime(start_ts))
            e = time.strftime(VIDEO_NAME_DT_FMT, time.localtime(end_ts))
        except (OSError, ValueError, OverflowError) as ex:
            raise SaveDataLayoutError(
                f"时间戳转视频文件名失败（start={start_ts}, end={end_ts}）: {ex}"
            ) from ex
        return f"{clip_idx:0{CLIP_IDX_WIDTH}d}-{s}-{e}_{kind}.{ext}"

    @staticmethod
    def build_video_path(videos_dir: str, clip_idx: int, start_ts: float,
                         end_ts: float, kind: str, ext: str = "mp4") -> str:
        """视频完整路径 = videos_dir / build_video_filename(...)。"""
        return os.path.join(
            videos_dir,
            SaveDataLayout.build_video_filename(clip_idx, start_ts, end_ts,
                                                kind, ext))

    # ------------------------------------------------------------------
    # 列表工具（供 N4 磁盘清理脚本复用）
    # ------------------------------------------------------------------
    def list_date_dirs(self) -> list:
        """列出 root 下所有合法日期目录名（YYYY-MM-DD），按名字升序。

        仅匹配 DATE_DIR_PATTERN 严格格式；其它名称（系统文件、临时目录、
        误建目录）一律忽略，保护 N4 清理脚本的「绝不删错」硬约束。
        """
        if not os.path.isdir(self.root):
            return []
        return sorted(
            name for name in os.listdir(self.root)
            if DATE_DIR_PATTERN.match(name)
            and os.path.isdir(os.path.join(self.root, name))
        )
