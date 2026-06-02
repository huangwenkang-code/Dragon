# Monster Matcher L3 + 妖股信号打分器 设计

2026-05-19

## 目标

两个独立模块，都挂在 `generate_candidates.py` 流程里：

1. **ChromaDB L3 文本语义层** — 用 sentence-transformers 对妖股 markdown 报告做 embedding，候选股以特征描述文本 query，输出语义相似度，与现有 L2 结构化特征 5:5 融合
2. **MonsterSignalScorer 规则打分器** — 基于已知妖股模式（小盘+游资+题材+情绪）的透明打分，输出 `ml_sub`，替换当前恒为 0

---

## 第一部分：ChromaDB L3 文本嵌入层

### 文件：`services/monster_matcher/embedding_store.py` (NEW)

### 模型

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 384 维向量，420MB，CPU <50ms/条
- 支持中文，无需 API key
- 惰性加载单例，只在首次使用时下载

### 数据

11 个 `data_lake/monsters/*.md` 文件，格式为 YAML frontmatter + Markdown 正文。
嵌入时去掉 YAML frontmatter，只取正文（从第二个 `---` 之后开始）。

### ChromaDB 存储

- Collection 名: `monster_reports`
- 持久化路径: `data_lake/chroma_db/monster_reports/`
- `is_persistent=True`，进程重启后数据不丢失
- 存储结构: `{doc_id: stock_code, document: markdown body, metadata: {stock_name, primary_type, sector, max_gain_pct, ...}}`
- 支持增量追加：已有 stock_code 的文档先删后插（upsert 语义）

### 匹配流程

1. **查询构造** — 将 `CandidateFeatures` 序列化为中文描述字符串：

```
候选股: {stock_name}({symbol}), 市值{market_cap}亿, 板块{sector}, 当前价格{price}元, 换手{change_pct}%
```

2. **Embedding** — 调用 sentence-transformers 生成 384 维向量
3. **ChromaDB query** — 返回 top-k 按余弦距离排序
4. **相似度转换** — `text_similarity = 1.0 - distance`（ChromaDB 返回的是余弦距离）

### 融合公式

```
final_similarity = 0.5 * structured_similarity + 0.5 * text_similarity
```

其中 `structured_similarity` 是现有 L2 的 7 维加权分数（市值/换手/价格/板块/类型/涨幅/标签）。

match_reasons 中附加文本匹配来源：如果 `text_similarity > 0.6`，追加"语义高度相似"的原因标签。

### 初始化时机

`MonsterEmbeddingStore` 单例在第一次调用 `find_similar()` 时懒初始化：
- 检查 ChromaDB collection 是否已有数据
- 有 → 直接使用
- 无 → 遍历 markdown 文件，逐篇 embedding 并写入

### 错误处理

- sentence-transformers 未安装 → 降级为纯 L2（text_similarity = 0.5，即不贡献区分度）
- ChromaDB 不可用 → 同降级
- 单篇 markdown 读取失败 → 跳过该篇，log warning

---

## 第二部分：MonsterSignalScorer 规则打分器

### 文件：`services/monster_matcher/signal_scorer.py` (NEW)

### 输入

从 `AgentState` 中提取的流水线数据：
- `capital_flow_records` — 市值、主力净流入、flow_score
- `dragon_tiger_records` — 龙虎榜、游资席位、trader_signal
- `sentiment_scores` — 情绪分、hype
- `events` — 事件题材
- `active_stocks` — matched_concepts

### 六个维度

| 维度 | 权重 | 信号逻辑 |
|------|------|---------|
| 市值规模 | 0.24 | <50亿=1.0, 50-100亿=0.7, 100-200亿=0.3, >200亿=0.0 |
| 游资参与 | 0.24 | famous_traders非空=1.0, 上榜无游资=0.5, 未上榜=0.0 |
| 情绪异常 | 0.19 | avg_sentiment>0.8 且 avg_hype>0.3=1.0, 仅sent>0.5=0.5, 否则0.2 |
| 题材热度 | 0.19 | matched_concepts>=3=1.0, 2个=0.7, 1个=0.4, 0个=0.0 |
| 连板趋势 | 0.14 | 基于 keyword 中含"连板""涨停"等词的数量估算，封顶 1.0 |

每个维度 0→1 线性或阶梯映射，不存在训练参数。

### 输出

- `score`: 0-1，加权总分
- `breakdown`: dict，每个维度的子分（用于审计/前端展示）

### 数据缺失降级

某个维度的数据源不可用时（如游资数据未收录），该维度记 0 分，其余维度权重按比例重分配。

---

## 集成点

### `generate_candidates.py` 修改

1. `_compute_scores()` — 将 `ml_score = model_scores.get(sym, 0.0)` 替换为调用 `MonsterSignalScorer.score(sym, ...)`
2. `_enrich_with_monster_reference()` — 现有只走 L2，改为同时走 L3 并融合
3. 移除 `_try_model_ensemble()` 及相关调用（保留函数但标记 deprecated，不再在正常流程中调用）
4. LeaderCandidate 返回中 `ml_sub` 填充实际值而非 0

### `matcher.py` 修改

- `MonsterMatcher.find_similar()` — 新增可选参数 `use_l3: bool = True`
- 融合逻辑在 `find_similar()` 内部完成，对外接口不变

### 导入清单

```
services/monster_matcher/
├── __init__.py        (existing)
├── matcher.py          (MODIFY — add L3 integration)
├── embedding_store.py  (NEW — ChromaDB + sentence-transformers)
└── signal_scorer.py    (NEW — rules-based monster signal scorer)
```

---

## 不做什么

- 不训练 ML 模型（样本不足，11 只妖股不够训练）
- 不引入 tick 级数据
- 不修改现有 DB schema（ChromaDB 不在 PostgreSQL 里，规则打分器不需要新列）
- 不修改前端（`ml_sub` 已经在详情面板展示，只是从 0 变成有意义的值）

---

## 验证清单

- [ ] 首次运行 — 自动下载模型 + 初始化 ChromaDB collection（11 篇）
- [ ] 二次运行 — 检测到已有 collection，跳过重建
- [ ] 候选股匹配 — 返回的 monster_reference 相似度提升（目标：top match >30%，之前 20-40%）
- [ ] match_reasons 出现"语义高度相似"标签（当 L3 分 >0.6 时）
- [ ] `ml_sub` 不再恒为 0，不同候选有区分度
- [ ] 无 GPU/离线环境 — sentence-transformers CPU 推理正常
- [ ] ChromaDB 不可用 — 降级为纯 L2，不崩溃
- [ ] 新增妖股 markdown — 运行一次追加脚本即可重建
