"""Event extraction prompt — the core cognitive engine for A-share event analysis.

Pattern: Biomni's monolithic sectioned system prompt (a1.py lines 1118-1310)
+ two-pass evaluateGPT output extraction (react.py lines 447-465).

IMPORTANT: System prompt uses SystemMessage(content=...) (static) to avoid
LangChain template brace-escaping hell. Only the human message uses template variables.
"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from services.cognitive_layer.prompts.domain_knowledge import COGNITIVE_DOMAIN_KNOWLEDGE

# ---------------------------------------------------------------------------
# Event extraction system prompt (static — no template variables)
# ---------------------------------------------------------------------------

EVENT_EXTRACTION_SYSTEM = f"""
{COGNITIVE_DOMAIN_KNOWLEDGE}

---

## 当前任务: 从新闻中提取结构化事件

输入格式: `[ID: n] 标题: xxx | 来源: xxx | 时间: xxx | 内容: xxx`

### 输出格式 (JSON数组)
```json
[
  {{
    "event_id": "evt_001",
    "event_type": "政策/产业/公告/突发/题材",
    "title": "事件标题(≤30字)",
    "summary": "2-3句话叙事摘要",
    "source": "新闻来源",
    "publish_time": "发布时间",
    "symbol_list": ["000001"],
    "sector_list": ["概念名称"],
    "sector_tags": ["题材标签"],
    "narrative": "什么事件→影响什么板块→预期持续多久",
    "event_strength": 0.85,
    "heat_score": 0.7,
    "keywords": ["关键词"],
    "novelty": 0.6,
    "scope": "sector"
  }}
]
```

约束:
- novelty: 全新0.8+, 更新0.3-0.5, 旧闻<0.2
- 新闻质量太差时返回 []
"""

# ---------------------------------------------------------------------------
# Human prompt template (has template variables: trade_date, news_count, news_text)
# ---------------------------------------------------------------------------

EVENT_EXTRACTION_HUMAN = """
以下是{trade_date}的A股板块/概念相关新闻, 共{news_count}条。

请按系统指令提取结构化事件:

{news_text}

以JSON数组输出。质量优先于数量。
"""

# ---------------------------------------------------------------------------
# Build ChatPromptTemplate
# ---------------------------------------------------------------------------

event_extraction_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=EVENT_EXTRACTION_SYSTEM),
    HumanMessagePromptTemplate.from_template(EVENT_EXTRACTION_HUMAN),
])
