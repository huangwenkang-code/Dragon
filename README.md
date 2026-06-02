# Dragon Engine

A股事件驱动龙头/妖股认知引擎 — Monorepo + Java/Python 混合架构

## 架构

```
dragon-engine/
├── services/          # 微服务
│   ├── graph-service/     # LangGraph 编排核心
│   ├── event-service/     # 事件抽取
│   ├── sentiment-service/ # 舆情/叙事分析
│   ├── feature-service/   # 特征工程
│   ├── model-service/     # 模型推理/训练
│   ├── risk-service/      # 风险拦截
│   ├── report-service/    # 报告生成
│   ├── data-adapter/      # 数据适配层
│   ├── llm-adapter/       # LLM 统一适配层
│   └── api-gateway/       # Java SpringBoot 网关
├── shared/            # 共享模块
│   ├── schemas/       # Pydantic 模型
│   ├── prompts/       # Prompt 模板
│   ├── configs/       # 配置
│   ├── memory/        # 记忆/向量存储
│   └── utils/         # 工具函数
├── infra/             # 基础设施
│   ├── docker/
│   ├── k8s/
│   └── scripts/
└── README.md
```

## 技术栈

- **Python**: 3.11, FastAPI, LangGraph, LangChain, Pydantic v2
- **Java**: 17, SpringBoot 3.x, Maven
- **Infra**: Redis, PostgreSQL, ChromaDB, Docker

## 快速启动

```bash
# 安装依赖
pip install -e .

# 启动 graph-service (最小可运行图)
cd services/graph-service
python main.py

# Docker Compose 全栈启动
docker-compose up -d
```

## 许可证

Apache 2.0
