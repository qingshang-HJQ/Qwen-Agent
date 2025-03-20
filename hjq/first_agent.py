"""
第一个agent,
"""

import os

from qwen_agent.agents import Assistant

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


def init_agent_service():
    llm_cfg = {
        'model': 'qwen-72b-chat',
        'model_type': 'qwen_dashscope',
        "system": """
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
        
        """,
        'generate_cfg': {
        }
    }

    tools = [

    ]
    bot = Assistant(
        llm=llm_cfg,
        function_list=tools,
        name='qwen-72b-chat Demo',
        description="qwen-72b-chat demo")

    return bot


def app():
    # Define the agent
    bot = init_agent_service()
    while True:
        query = input('user question: ')
        messages = [{'role': 'user', 'content': query}]
        # response = bot.run(messages=messages)
        # a = ""
        # for i in response:
        #     a += i[0]['content']
        # print(a)
        for response in bot.run(messages=messages):
            print('bot response:', response)




if __name__ == '__main__':
    app()