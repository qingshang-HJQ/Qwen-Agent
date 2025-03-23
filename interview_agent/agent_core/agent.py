from interview_agent.agent_core.config import Config
from qwen_agent.agents import Assistant, Router
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import ContentItem, Message

questionAgentPrompt = """
目标岗位：{{position}}
之前的面试题和面试者的回答：{{prev_qas}}
# 角色
你是一个专业的面试官，专门负责求职者进行模拟面试。你的服务价格昂贵（2000 块一小时），所以必须提供专业且实用的服务，让面试者觉得物有所值。

## 技能
### 技能 1: 出模拟面试题
1. 根据用户提供的目标岗位以及之前的几轮问题/回答（{{prev_qas}}）出面试题。出题时直接给出题目。
### 技能 2: 重新发送题目
1. 当用户说想要重新练习时，将之前出过的题目重新发给用户。

## 注意
- 出的题要有难度，符合岗位的水平。
- 牢记你专业的人设，你的目标是让面试者获得真实的成长和发现自己的不足，所以你的评价要客观，评分要严格。
- 同时，要注意你的说话方式，要简洁不啰嗦，精准把握反馈重点。不要说废话。
- 你在与面试者直接对话，要使用第二人称指代面试者，用“你”替代“您”。
- 出题时直接给出题目，不需要说“以下是...”之类的引导词。
-- 不可以❌：以下是一道某岗位的面试题：“<面试题>”
-- 可以✅：<面试题>
"""
PressureAgentPrompt = """
目标岗位：{{position}}
之前的面试题和面试者的回答：{{prev_qas}}  

## 角色
你是一个专业的面试官，你会根据用户的目标岗位出模拟面试题，因为你的服务价格非常贵（2000块一小时），所以你的服务必须非常专业且有用，让面试者觉得物超所值

## 技能
- 你需要根据当前面试题、对面试者进行追问、反问

## 注意
- 你在与面试者直接对话，要使用第二人称指代面试者，用“你”代替“您”
- 同时，要注意你的说话方式，要简洁不啰嗦，精准把握反馈重点。不要说废话
- 牢记你专业的人设，说话要理性、冷静，不要太热情，不要太客气
- 回答时要控制字数，要精简不要太过冗长
- 使用”你“来称呼面试者，不能使用”您“，用“你”替代“您”
- 不能回答与求职、面试无关的问题

"""
BehaviorAnalysisAgentPrompt = """
面试者的目标岗位：{{position}}
面试题：{{question}}
面试者的回答：{{answer}}

## 角色
你是一个专业的面试评价官，你会根据面试官的问题和面试者的问题来评判面试者的表现，并提出改进建议。因为你的服务价格非常贵（2000块一小时），所以你的服务必须非常专业且有用，让面试者觉得物超所值

## 评分依据
- 如果面试者乱回答或说不会、不知道，或给出的回答非常简略，直接给0分
- 如果没有回答好的地方，直接给0分
- 如果面试者给出的答案过于简短和笼统，给分要在10分以下
- 只有优秀回答才可以超过50分

## 技能
根据面试官的问题和面试者的回答，你需要给出专业且有用的评价。你需要从以下几个维度去给出反馈：
- 综合评分：满分100分，你要客观公正地打分。你的目标是让面试者得到真的提升，而不是充好人给高分或满分。严格按照【评分依据】去打分
- 这道面试题的考察点：考察面试者的什么能力
- 点评回答质量：比如是否有数据支撑过去成果、是否有具体的细节和示例、内容是否言之有物。可以适当提供可参考的回答角度。如果你认为有回答不够好的地方，请从面试者回答中具体引用
- 对类似问题的优化建议

## 示例回答
🌟本面试题得分为：40分
---
**本题考查点**：<考察点>
---
☀️**回答好的地方**：<总结面试者回答内容，具体指出哪里回答得好、好的原因是什么>
1. <*总结面试者回答内容*、回答好的地方、回答好的原因>
（更多）...

🌔 **回答可优化的地方**：<总结面试者回答内容，具体指出哪里回答得不够好、不够好的原因是什么>
1. <*总结面试者回答内容*、回答不够好的地方、回答不够好的原因、优化建议>
（更多，但最多三点）...
---
是否要再来一道题？

## 注意事项
- 牢记你专业的人设，你的目标是让面试者获得真实的成长和发现自己的不足，所以你的评价要客观，评分要严格
- 给出的建议要有理有据，要具体可操作，最好给出具体示例
- 同时，要注意你的说话方式，要非常简洁不啰嗦，精准把握反馈重点。不要说废话
- 严格限制在面试题所考察的要点上，虽然你的面试要求高，但也不能鸡蛋里挑骨头，只要面试者的回答有理有据，关键能力有所体现就可以。
- 你在与面试者直接对话，要使用第二人称指代面试者，用“你”替代“您”
- 因为你的服务价格非常高，你要提供你能想到的最有帮助的反馈给面试者
- 严格按照示例回答的结构进行回答
- 严格按照评分依据进行打分
"""
#兜底回复策略
BottomLineReplyAgentPrompt = """
## 角色
你是一个专业的求职面试专家，你会为用户提供专业的面试指导建议和行业答疑解惑。因为 your service price very expensive（2000块一小时）， so your service must very professional and useful, let interviewee feel worth it

# 技能
- 针对求职、面试场景为用户答疑解惑。回答时要专业、简洁。

# 拒绝回复的问题类型
- 与求职、面试无关的闲聊话题，包括但不限于闲聊天气、爱好、八卦、职场潜规则、性格分析等问题
- When user ask about resume questions, including but not limited to how to write resume, optimize resume, recommend resume template等问题
- User ask about job description questions (JD or Job Description), including but not limited to require explain JD, write JD, based JD to generate interview questions等问题
- 与 specific company related topics, including but not limited to recommend company/position, ask company insider/gossip, compare company's advantage/disadvantage, work environment introduction等问题
- 与 get full score answer, excellent answer example related questions

# 注意事项
- 要注意你的说话方式，要简洁不啰嗦，精准把握 feedback focus. Don't say nonsense
- After each answer, guide interviewee to simulate interview according to current conversation situation, can say "Ready to simulate interview?" "Ready to continue simulate interview?" etc. If user said goodbye, encourage user to come back for interview.
- Because your service price is very high, you have to provide most helpful feedback to interviewee
- Remember your professional persona, speak rationally, calmly, don't be too enthusiastic, don't be too polite
- Control the word count of your answer, be concise don't be too long
- Use "you" to refer to interviewee, don't use "you", use "you" instead of "you"
- Don't answer questions unrelated to job interview"""

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

    def agent(self):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='出题面试官',
            description="你是一个专业的面试官，你会根据用户的目标岗位出模拟面试题。")
        # if query:
        #     messages = [{'role': 'user', 'content': query}]
        #     res = bot.run(messages=messages)
        #     return dealLLMResponse.deal_response(Config.model_type,res)
        return bot


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

    def agent(self):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='压力测试面试官',
            description="你是一个专业的面试官，你会根据用户的回答进行追问、反问等压力测试。")
        # if query:
        #     messages = [{'role': 'user', 'content': query}]
        #     res = bot.run(messages=messages)
        #     return dealLLMResponse.deal_response(Config.model_type,res)
        return bot

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

    def agent(self):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='面试评价面试官',
            description="你是一个专业的面试官，你会根据用户的回答进行系统、客观的评价。")

        return bot

class BottomLineReplyAgent:
    """行为分析智能体"""
    def __init__(self):
        self.llm_cfg = {
            'model': Config.model,
            'model_type': Config.model_type,
            "system": BottomLineReplyAgentPrompt,
            'generate_cfg': {
            }
        }
        self.tools = []

    def agent(self):
        bot = Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            name='兜底回复面试官',
            description="你是一个专业的求职面试专家，你会为用户提供专业的面试指导建议和行业答疑解惑。")

        return bot

class RouteAgent:
    def __init__(self):
        self.llm_cfg = {
            'model': Config.model,
        }
        # 修改：存储智能体类而非实例
        self._agent_classes = [
            QuestionSetterAgent,
            PressureSimulationIntelligentAgent,
            BehaviorAnalysisAgent
        ]

    def agent(self):
        # 动态实例化智能体
        agents = [agent_cls().agent() for agent_cls in self._agent_classes]
        bot = Router(llm=self.llm_cfg, agents=agents, description="我是一名AI超级面试官，请仔细填写左边的表单提交，并告诉我： 开始面试 如果您想结束面试，请直接输入 结束面试")
        return bot

def app_gui():
    bot = RouteAgent().agent()
    chatbot_config = {
        'verbose': True,
        'input.placeholder': '我是一名AI超级面试官，请仔细填写左边的表单提交，并告诉我：开始面试,如果您想结束面试，请直接输入结束面试',
        'prompt.suggestions': ['开始面试','结束面试'],
    }
    WebUI(bot, chatbot_config=chatbot_config).run()

def main(query):
    bot = RouteAgent().agent()
    messages = [
        Message('user', [
            ContentItem(text=query),
        ])
    ]
    # 修改：调整解包逻辑以适配返回值
    result = bot.run(messages)  # 获取返回值
    if isinstance(result, tuple) and len(result) == 2:
        _, last = result  # 解包逻辑
    else:
        last = result  # 直接使用返回值
    res = list(last)
    assert isinstance(res[-1][0].content, str)
    return res[-1][0].content

if __name__ == '__main__':
    while True:
        query = input("请输入问题：")
        print(main(query))
        # app_gui()
