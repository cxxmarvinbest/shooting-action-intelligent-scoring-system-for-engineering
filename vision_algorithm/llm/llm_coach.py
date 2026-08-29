# -*- coding: utf-8 -*-
"""
API 调用大语言模型评分模块（llm/llm_coach）
==============================================
职责：封装通过 API 调用大语言模型（豆包 Doubao）生成教练评语的功能。
  1. 根据各评分项组装提示词（Prompt）
  2. 调用 Ark 对话补全接口
  3. 解析并返回评语文本（含离线模式 / 异常兜底）

离线模式（OFFLINE）：
  - 开启后【完全不发起网络请求】，改用本地规则基于六项评分生成评语，
    用于无网络 / 断网测试，保证流程不报错、不阻塞。
  - 开关：构造 LLMCoach(offline=True) 或环境变量 LQ_OFFLINE=1。

异常兜底：
  - 在线模式若发生网络异常（断网 / 超时 / DNS 失败 / 非 200 等），
    自动降级为本地规则评语，不会抛出、不会长时间阻塞（超时默认 30s）。

对外只暴露：LLMCoach
依赖：requests / config（API 配置见 config/llm.yaml）
"""

import logging
import os

import requests

from config import Config

logger = logging.getLogger("basketball_scoring")


class LLMCoach:
    """大语言模型教练评语生成器（职责单一：只负责"问 AI 要评语"）"""

    # ── 火山引擎 Ark API 配置（见 config/llm.yaml）──
    API_URL = Config.LLM_API_URL
    API_KEY = Config.LLM_API_KEY
    MODEL_ID = Config.LLM_MODEL_ID

    # 在线请求超时（秒）：断网/弱网时尽快失败，避免主流程长时间阻塞
    REQUEST_TIMEOUT = Config.LLM_TIMEOUT

    # 极端兜底（仅在本地规则也不可用时，原则上不会走到）
    FALLBACK_REPORT = "AI 教练评语生成失败，但评分流程已完成。"

    def __init__(self, offline=False):
        # offline 可由构造参数 / 环境变量 LQ_OFFLINE=1 / config.llm.yaml 默认值控制
        self.offline = offline or bool(int(os.environ.get("LQ_OFFLINE", "0"))) \
            or Config.LLM_OFFLINE_DEFAULT

    def build_prompt(self, score1, score2, completeness_score,
                     coord_score, knee_score, release_score):
        """根据六项评分数据组装教练提示词"""
        return f"""
你是一位专业的篮球教练。请根据以下我的投篮测试数据，给我一份简短、专业的中文改进建议：
1. 准备下蹲阶段动作相似度得分：{score1:.1f}/100
2. 蹬伸出手阶段动作相似度得分：{score2:.1f}/100
3. 核心技术环节完整度：{completeness_score:.1f}%
4. 动力链协同发力得分：{coord_score:.1f}/100
5. 屈膝爆发力得分：{knee_score:.1f}/100
6. 出手角度得分：{release_score:.1f}/100
[输出格式要求]
必须直接输出以下三段内容，禁止任何自我介绍或"收到"、"好的"等客套话，总字数控制在 150 字左右：
1.亮点：[选取完整度、动力链协同、屈膝爆发力、出手角度中得分最高 1‑2 项，肯定该技术环节完成质量，动作流程表现稳定]
2.问题：[部分技术维度分数偏低，存在动作衔接断档，发力链条传递存在缺陷，部分关键动作执行质量有待提升]
3.建议：进行[具体的辅助训练]练习/辅助练习，加入[具体的修正细节]动作，体会[具体身体部位或技术环节]的发力。
"""

    def _build_local_report(self, score1, score2, completeness_score,
                            coord_score, knee_score, release_score,
                            reason="offline"):
        """
        离线 / 网络异常时的本地规则化评语（无网络依赖）。

        reason:
            "offline"       -> 离线模式主动跳过联网
            "network_fail"  -> 在线模式尝试联网但失败，降级而来
        返回：可直接展示的评语文本。
        """
        # 统一到 0~100 量级做极值比较（完整度为百分比，其余为 /100，量纲一致）
        items = [
            ("准备下蹲阶段动作相似度", score1),
            ("蹬伸出手阶段动作相似度", score2),
            ("核心技术环节完整度", completeness_score),
            ("动力链协同发力", coord_score),
            ("屈髋屈膝发力", knee_score),
            ("出手角度", release_score),
        ]
        best_name, best_val = max(items, key=lambda kv: kv[1])
        worst_name, worst_val = min(items, key=lambda kv: kv[1])

        if reason == "network_fail":
            header = "【豆包接口暂不可用 · 本地自动评语】"
            note = "（豆包大模型接口调用失败，已自动降级为基于评分的本地规则分析）"
        else:
            header = "【离线模式 · 本地自动评语】"
            note = "（未接入豆包大模型，以上为基于评分的本地规则分析，仅供参考）"

        lines = [header,
                 f"亮点：{best_name}表现最好（{best_val:.1f}/100），请保持该环节的技术稳定性。"]
        if worst_val < 75.0:
            lines.append(
                f"待改进：{worst_name}偏低（{worst_val:.1f}/100），建议针对性加强该环节的训练与纠正。")
        else:
            lines.append(
                f"整体：各项指标均达合格线（最低为{worst_name} {worst_val:.1f}/100），"
                f"动作完成度较好，可继续打磨细节。")
        lines.append(note)
        return "\n".join(lines)

    def generate_report(self, score1, score2, completeness_score,
                        coord_score, knee_score, release_score,
                        offline=None):
        """
        生成教练评语。

        参数：
            score1/score2/completeness_score/coord_score/knee_score/release_score：六项评分
            offline：
                None -> 使用 self.offline（构造时的开关 / 环境变量）
                True -> 强制离线，不发起网络请求
                False -> 强制在线
        返回：评语文本。离线或异常时返回本地规则评语，绝不抛出。
        """
        use_offline = self.offline if offline is None else offline
        if use_offline:
            return self._build_local_report(score1, score2, completeness_score,
                                            coord_score, knee_score, release_score,
                                            reason="offline")

        ai_prompt = self.build_prompt(score1, score2, completeness_score,
                                      coord_score, knee_score, release_score)
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.API_KEY}"
            }
            payload = {
                "model": self.MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": ai_prompt, "type": "text"}]
                    }
                ]
            }
            response = requests.post(self.API_URL, headers=headers,
                                     json=payload, timeout=self.REQUEST_TIMEOUT)
            if response.status_code == 200:
                resp_json = response.json()
                return resp_json['choices'][0]['message']['content']
            # 非 200：接口可用但业务出错，降级为本地评语
            logger.warning("豆包 API 返回非 200 状态码 %s（响应：%s），降级为本地评语",
                           response.status_code, response.text[:200])
            return self._build_local_report(score1, score2, completeness_score,
                                            coord_score, knee_score, release_score,
                                            reason="network_fail")
        except Exception as e:
            # 断网 / 超时 / DNS 失败等任意异常，均降级为本地评语，不抛出。
            # 记录具体异常类型便于部署时定位（DNS 失败 / 连接超时 / 证书问题等）。
            logger.warning("豆包 API 调用异常（%s: %s），降级为本地评语",
                           type(e).__name__, e)
            return self._build_local_report(score1, score2, completeness_score,
                                            coord_score, knee_score, release_score,
                                            reason="network_fail")
