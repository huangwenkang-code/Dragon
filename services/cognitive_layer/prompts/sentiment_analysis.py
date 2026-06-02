"""Sentiment analysis prompt — evaluates multi-dimensional sentiment for A-share events.

Pattern: Biomni evaluateGPT two-pass (extract events first, then analyze sentiment).
"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from services.cognitive_layer.prompts.domain_knowledge import SENTIMENT_ANALYSIS_RULES

# ---------------------------------------------------------------------------
# Sentiment analysis system prompt (static — no template variables)
# ---------------------------------------------------------------------------

SENTIMENT_ANALYSIS_SYSTEM = f"""
你是一位A股市场情绪分析专家，专门评估股票在事件驱动下的多维度情绪。

{SENTIMENT_ANALYSIS_RULES}

---

## 当前任务: 对给定的事件+资金流向数据, 为每只受益标的生成多维度情绪评分

### 输入信息
你会收到三部分数据:
1. **事件列表**: 今天从板块/概念新闻中提取的结构化事件
2. **资金流向**: 相关股票的主力资金流向数据(主力净流入、大单净量、成交额)
3. **龙虎榜**: 如果有龙虎榜数据, 包含游资席位和买卖情况

### 分析逻辑
对于每只股票, 综合以下信号:
- 事件强度 × 相关性 = 事件驱动力
- 主力资金方向 × 力度 = 资金确认度
- 游资信号(合力/分歧/出货) = 短期动能
- 多源一致性 = 可信度

### 输出格式

```json
[
  {{
    "target_id": "000001",
    "target_type": "stock",
    "symbol": "000001",
    "sentiment_score": 0.65,
    "narrative_score": 0.8,
    "hype_score": 0.4,
    "consistency_score": 0.75,
    "risk_score": 0.2,
    "confidence": 0.7,
    "heat": 0.55,
    "consensus": 0.75,
    "diffusion_speed": 0.6,
    "narrative_strength": 0.8,
    "keywords": ["龙头", "主力流入", "板块联动"]
  }}
]
```

### 关键约束
- sentiment_score必须在[-1, 1]范围内, 其他分数在[0, 1]范围内
- 如果没有足够信息支撑分析, confidence<0.5
- 资金数据是最客观的信号, 应给予较高权重
- 识别"利好出尽"模式: 事件强度高但资金流出 → 风险信号
- 识别"利空出尽"模式: 坏消息但资金逆势流入 → 可能见底
"""

# ---------------------------------------------------------------------------
# Human prompt template
# ---------------------------------------------------------------------------

SENTIMENT_ANALYSIS_HUMAN = """
以下是{trade_date}的数据:

## 事件列表 (共{event_count}条)
{events_text}

## 资金流向 (共{flow_count}条)
{flow_text}

## 龙虎榜 (共{lhb_count}条)
{lhb_text}

请对每只受益标的生成六维情绪评分。重点分析:
1. 事件驱动与资金流向是否一致(一致=高可信度)
2. 是否有"利好出尽"或"见光死"风险
3. 板块内是否有扩散效应(多只股票同时获得资金流入)
"""

# ---------------------------------------------------------------------------
# Build ChatPromptTemplate
# ---------------------------------------------------------------------------

sentiment_analysis_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SENTIMENT_ANALYSIS_SYSTEM),
    HumanMessagePromptTemplate.from_template(SENTIMENT_ANALYSIS_HUMAN),
])
