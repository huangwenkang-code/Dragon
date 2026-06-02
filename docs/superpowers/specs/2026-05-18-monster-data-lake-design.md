# 妖股数据湖设计

## 概述

为 dragon-engine 认知引擎建立妖股案例库（Case-Based Reasoning），让 LLM 在做候选生成时可以对比历史相似妖股，而不是凭空推理。

## 架构

### 三层存储

| 存储层 | 内容 | 用途 |
|--------|------|------|
| PostgreSQL: `monster_stock` | 妖股元信息 + YAML 结构化字段 | L1 精确过滤 |
| PostgreSQL: `monster_daily_bar` + `monster_minute_bar` | 日线/分钟线 OHLCV + phase 标签 | L2 特征向量计算排序 |
| `data_lake/monsters/*.md` | 叙事文本 + 事件链 + 特征总结 | L3 LLM 类比推理生成 |
| AKShare 实时 | 候选股近 30 天日线 | 在线推理时抽取特征 |

### 检索流程

```
候选股 → 元数据过滤(L1) → 日线特征向量排序(L2) → 分钟线精细特征微调(L2) → Markdown叙事检索(L3) → LLM生成类比
```

## 数据库表

### monster_stock

CREATE TABLE monster_stock (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(20) NOT NULL,
    primary_type VARCHAR(30) NOT NULL,
    secondary_type VARCHAR(30) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    trading_days INT,
    start_price NUMERIC(10,3),
    peak_price NUMERIC(10,3),
    max_gain_pct NUMERIC(8,2),
    drawdown_pct NUMERIC(8,2),
    market_cap_start NUMERIC(12,2),
    market_cap_peak NUMERIC(12,2),
    turnover_avg_pre NUMERIC(6,2),
    turnover_avg_surge NUMERIC(6,2),
    limit_up_count INT,
    sector VARCHAR(30),
    markdown_path VARCHAR(500),
    similar_cases TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

### monster_daily_bar / monster_minute_bar

日线和分钟线表结构见 spec 内联 SQL。phase 字段由自动化脚本自动标注：
- pre_accumulate: start_date 前 15 个交易日
- early_surge: 主升浪前 1/3
- mid_surge: 主升浪中间 1/3
- peak: 最后几个交易日
- decline: end_date 后 10 个交易日

## 分类体系

### monster_type 分类

第一维度（核心驱动力）：theme_speculation / policy_driven / industry_trend / earnings_driven / restructuring / ipo_speculation / turnaround / sentiment_leader

第二维度（时间跨度）：short_term / swing / long_term / cross_year

组合使用，如 `earnings_driven + long_term`（九安医疗）、`theme_speculation + short_term`（捷荣技术）。

## Markdown 模板

每个妖股一个 markdown 文件，包含：
1. YAML frontmatter（结构化元数据，机器解析）
2. 基本面画像（启动前/峰值对比表）
3. 主升浪时间线（吸筹期/主升各阶段/见顶期）
4. 核心驱动事件链（影响链分析）
5. 资金面深度（龙虎榜行为模式表）
6. 情绪/叙事演变（阶段叙事变化）
7. 关键特征总结（启动前/主升浪/见顶各阶段特征）

## 文件组织

dragon-engine/
├── data_lake/
│   ├── monsters/                  # 妖股 markdown
│   │   ├── 002855_捷荣技术.md
│   │   ├── 002432_九安医疗.md
│   │   └── ...
│   ├── daily_bars/               # CSV 冷备
│   └── import.py                 # 导入工具
├── services/
│   └── monster_matcher/
│       ├── __init__.py
│       ├── matcher.py
│       └── feature_extractor.py
└── db/
    └── schema.sql

## 实施计划

### P0: 最小可行验证
- 创建 3 张表 (monster_stock, monster_daily_bar, monster_minute_bar)
- 手选 10-15 只代表性妖股，覆盖全类型
- 生成高质量 markdown（LLM辅助 + 人工校对）
- 导入 Postgres + ChromaDB
- 基础相似度检索验证

### P1: 结构化特征混合
- 实现 feature_extractor（从日线/分钟线抽取特征向量）
- 混合文本 + 结构化特征检索
- 验证 recall@5 是否显著优于纯文本

### P2: 候选股实时对比
- 集成到 generate_candidates 节点
- 在线推理时实时拉取候选股特征，匹配历史案例
- LLM 生成类比分析文本
