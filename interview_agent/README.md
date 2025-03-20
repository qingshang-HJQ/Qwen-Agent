├── agent_core/
│   ├── __init__.py
│   ├── agent.py          # agent class
│   ├── planning.py          # 规划模块，负责任务分解和决策
│   ├── memory.py           # 记忆模块，负责存储和检索信息
│   ├── tools.py            # 工具模块，负责与外部系统交互
│   ├── action.py           # 动作模块，负责执行具体任务
│   └── main.py            # 主程序入口，协调各模块工作
├── data/
│   ├── raw_data/           # 原始数据存储
│   ├── processed_data/     # 处理后的数据存储
│   └── models/             # 训练好的模型文件
├── logs/                   # 日志文件存储
├── tests/                  # 单元测试和集成测试代码
│   ├── test_planning.py
│   ├── test_memory.py
│   ├── test_tools.py
│   └── test_action.py
├── requirements.txt        # 项目依赖包列表
├── setup.py                # 项目安装和配置文件
└── README.md               # 项目说明文档