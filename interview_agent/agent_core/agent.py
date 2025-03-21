from interview_agent.agent_core.config import Config
from qwen_agent.agents import Assistant

questionAgentPrompt = """
# 角色
你是一个专业的面试评价官，专门负责对有 3 年左右工作经验的 Python 后端开发岗位求职者进行模拟面试。你的服务价格昂贵（2000 块一小时），所以必须提供专业且实用的服务，让面试者觉得物有所值。

## 技能
### 技能 1: 出模拟面试题
1. 根据用户提供的目标岗位（3 年左右工作经验的 Python 后端开发岗位）以及之前的几轮问题/回答（{{prev_qas}}）出面试题。出题时直接给出题目。
### 技能 2: 重新发送题目
1. 当用户说想要重新练习时，将之前出过的题目重新发给用户。

## 注意
- 出的题要有难度，符合 3 年左右工作经验的 Python 后端开发岗位的水平。
- 牢记你专业的人设，你的目标是让面试者获得真实的成长和发现自己的不足，所以你的评价要客观，评分要严格。
- 同时，要注意你的说话方式，要简洁不啰嗦，精准把握反馈重点。不要说废话。
- 你在与面试者直接对话，要使用第二人称指代面试者，用“你”替代“您”。
- 出题时直接给出题目，不需要说“以下是...”之类的引导词。
-- 不可以❌：以下是一道 3 年左右工作经验的 Python 后端开发岗位的面试题：“<面试题>”
-- 可以✅：<面试题>
"""
PressureAgentPrompt = """"""

class dealLLMResponse:
    @staticmethod
    def deal_response(LLMType,response):
        res_msg = ""
        if LLMType == Config.model_type:
            for _res in response:
                if _res[0]["extra"]["model_service_info"]["output"]["choices"][0]["finish_reason"] == "stop":
                    res_msg += _res[0]["content"]
                    break
        return res_msg


class QuestionSetterAgent:
    "出题面试管"

    def __init__(self):
        self.llm_cfg = {
            'model': Config.model,
            'model_type': Config.model_type,
            "system": questionAgentPrompt,
            'generate_cfg': {
            }
        }
        self.tools = []

    def agent(self,query):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='qwen-72b-chat Demo',
            description="qwen-72b-chat demo")
        if query:
            messages = [{'role': 'user', 'content': query}]
            res = bot.run(messages=messages)
            return dealLLMResponse.deal_response(Config.model_type,res)
        return None


class PressureSimulationIntelligentAgent:
    """压力测试智能体"""
    def __init__(self):
        self.llm_cfg = {
            'model': Config.model,
            'model_type': Config.model_type,
            "system": PressureAgentPrompt,
            'generate_cfg': {
            }
        }
        self.tools = []

    def agent(self,query):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='qwen-72b-chat Demo',
            description="qwen-72b-chat demo")
        if query:
            messages = [{'role': 'user', 'content': query}]
            res = bot.run(messages=messages)
            return dealLLMResponse.deal_response(Config.model_type,res)
        return None

class BehaviorAnalysisAgent:
    """行为分析智能体"""
    def __init__(self):
        self.llm_cfg = {
            'model': Config.model,
            'model_type': Config.model_type,
            "system": questionAgentPrompt,
            'generate_cfg': {
            }
        }
        self.tools = []

    def agent(self,query):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='qwen-72b-chat Demo',
            description="qwen-72b-chat demo")
        if query:
            messages = [{'role': 'user', 'content': query}]
            res = bot.run(messages=messages)
            return dealLLMResponse.deal_response(Config.model_type,res)
        return None


if __name__ == '__main__':
    question_setter = QuestionSetterAgent()
    print(question_setter.agent("请你出题面试，我们开始面试吧"))
