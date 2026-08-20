# -*- coding: utf-8 -*-
"""
API 调用大语言模型评分模块（llm_api）
======================================
职责：封装通过 API 调用大语言模型（豆包 Doubao）生成教练评语的功能。
  1. 根据各评分项组装提示词（Prompt）
  2. 调用 Ark 对话补全接口
  3. 解析并返回评语文本（含异常兜底）

对外只暴露：LLMCoach
依赖：requests（不依赖检测、评分、UI 模块）
"""

import requests


class LLMCoach:
    """大语言模型教练评语生成器（职责单一：只负责"问 AI 要评语"）"""

    # ── 火山引擎 Ark API 配置 ──
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    API_KEY = "ark-c661ccb2-f3af-4e66-a6b3-cf51d448206e-06359"
    MODEL_ID = "ep-m-20260720143111-jtzg2"

    # 兜底文案（网络异常 / 请求失败时的默认返回）
    FALLBACK_REPORT = "AI 教练开小差了，未能生成报告。"

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
1.亮点：[结合得分最高的1-2个指标，夸奖动作做得好的地方]
2.问题：[结合得分偏低或不达标的指标，指出动作中的断档或核心硬伤]
3.建议：进行[具体的辅助训练]练习/辅助练习，加入[具体的修正细节]动作，体会[具体身体部位或技术环节]的发力。
"""

    def generate_report(self, score1, score2, completeness_score,
                        coord_score, knee_score, release_score):
        """
        调用大语言模型生成教练评语。

        参数：六项评分数据
        返回：评语文本（失败时返回兜底文案或错误信息）
        """
        ai_prompt = self.build_prompt(score1, score2, completeness_score,
                                      coord_score, knee_score, release_score)

        ai_report = self.FALLBACK_REPORT
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
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=360)
            if response.status_code == 200:
                resp_json = response.json()
                ai_report = resp_json['choices'][0]['message']['content']
        except Exception as e:
            ai_report = f"AI 请求失败: {str(e)}"

        return ai_report
