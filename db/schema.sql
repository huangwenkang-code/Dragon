-- ============================================================================
-- dragon-engine PostgreSQL Schema v2
-- 对齐 7-node LangGraph 管线实际输出字段 (2026-05-17)
-- 设计原则: 每表对应一个管线节点产出, 最小化 JOIN, 索引覆盖核心查询
-- ============================================================================

-- 0. 股票基础信息 (reference data, 同花顺/tx_finance 缓慢变化)
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_basics (
    stock_code      VARCHAR(10) PRIMARY KEY,
    stock_name      VARCHAR(50) NOT NULL,
    industry        VARCHAR(50) DEFAULT '',
    market_cap      DOUBLE PRECISION DEFAULT 0,       -- 总市值(亿)
    circulating_cap DOUBLE PRECISION DEFAULT 0,       -- 流通市值(亿)
    pe              DOUBLE PRECISION DEFAULT 0,
    pb              DOUBLE PRECISION DEFAULT 0,
    list_date       VARCHAR(10) DEFAULT '',
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 1. 管道运行记录 (每次 POST /run 一条)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          VARCHAR(50) PRIMARY KEY,           -- ISO timestamp: 2026-05-17T09:30:00
    trade_date      DATE NOT NULL,
    status          VARCHAR(20) DEFAULT 'completed',   -- completed / partial / failed
    watchlist       TEXT[] DEFAULT '{}',
    top_n           INT DEFAULT 5,
    event_count     INT DEFAULT 0,
    candidate_count INT DEFAULT 0,
    top_score       DOUBLE PRECISION DEFAULT 0,
    errors          TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_runs_trade_date ON pipeline_runs (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON pipeline_runs (created_at DESC);

-- 2. 事件 (ingest_event 产出)
-- ============================================================================
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    event_id        VARCHAR(50) DEFAULT '',
    event_type      VARCHAR(20) DEFAULT '',            -- 政策/产业/公告/突发/题材
    title           TEXT NOT NULL,
    summary         TEXT DEFAULT '',
    content         TEXT DEFAULT '',
    source          VARCHAR(100) DEFAULT '',
    publish_time    VARCHAR(30) DEFAULT '',
    narrative       TEXT DEFAULT '',
    event_strength  DOUBLE PRECISION DEFAULT 0,        -- 0-1 (event_strength)
    heat_score      DOUBLE PRECISION DEFAULT 0,        -- 0-1
    strength        DOUBLE PRECISION DEFAULT 0,        -- 0-1 (备用)
    novelty         DOUBLE PRECISION DEFAULT 0,        -- 0-1
    scope           VARCHAR(20) DEFAULT 'individual',  -- individual/sector/market
    keywords        TEXT[] DEFAULT '{}',
    sector_list     TEXT[] DEFAULT '{}',               -- 事件涉及板块
    sector_tags     TEXT[] DEFAULT '{}',               -- 同花顺题材标签 (概念名列表)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_run_id      ON events (run_id);
CREATE INDEX IF NOT EXISTS idx_events_event_type  ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_publish_time ON events (publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_keywords    ON events USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_events_sector_tags ON events USING GIN (sector_tags);

-- 事件-股票关联 (多对多, 来自 event.symbol_list)
-- ============================================================================
CREATE TABLE IF NOT EXISTS event_stocks (
    id          SERIAL PRIMARY KEY,
    event_id    INT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    stock_code  VARCHAR(10) NOT NULL,
    UNIQUE (event_id, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_es_event ON event_stocks (event_id);
CREATE INDEX IF NOT EXISTS idx_es_stock ON event_stocks (stock_code);

-- 3. 情绪评分 (analyze_sentiment 产出)
-- ============================================================================
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id                SERIAL PRIMARY KEY,
    run_id            VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    target_id         VARCHAR(50) DEFAULT '',
    target_type       VARCHAR(20) DEFAULT 'stock',     -- stock / sector / event
    symbol            VARCHAR(10) DEFAULT '',
    sentiment_score   DOUBLE PRECISION DEFAULT 0,      -- -1 ~ 1
    narrative_score   DOUBLE PRECISION DEFAULT 0,      -- 0-1
    hype_score        DOUBLE PRECISION DEFAULT 0,      -- 0-1
    consistency_score DOUBLE PRECISION DEFAULT 0,      -- 0-1
    risk_score        DOUBLE PRECISION DEFAULT 0,      -- 0-1
    confidence        DOUBLE PRECISION DEFAULT 0.5,    -- 0-1
    heat              DOUBLE PRECISION DEFAULT 0,
    consensus         DOUBLE PRECISION DEFAULT 0.5,
    diffusion_speed   DOUBLE PRECISION DEFAULT 0,      -- 0-1
    narrative_strength DOUBLE PRECISION DEFAULT 0,     -- 0-1
    keywords          TEXT[] DEFAULT '{}',
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sent_run_id  ON sentiment_scores (run_id);
CREATE INDEX IF NOT EXISTS idx_sent_symbol ON sentiment_scores (symbol);
CREATE INDEX IF NOT EXISTS idx_sent_target ON sentiment_scores (target_id, target_type);

-- 4. 资金流向 (capital_flow + merge_active_stocks 产出)
--    字段完整覆盖: tx_finance 基础数据 + 同花顺 ddejingliang + EastMoney 分拆
-- ============================================================================
CREATE TABLE IF NOT EXISTS capital_flow_records (
    id                SERIAL PRIMARY KEY,
    run_id            VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    symbol            VARCHAR(10) NOT NULL,
    stock_name        VARCHAR(50) DEFAULT '',
    -- 行情
    price             DOUBLE PRECISION DEFAULT 0,      -- 最新价
    change_pct        DOUBLE PRECISION DEFAULT 0,      -- 涨跌幅%
    amount            DOUBLE PRECISION DEFAULT 0,      -- 成交额(元)
    amount_wan        DOUBLE PRECISION DEFAULT 0,      -- 成交额(万元)
    -- 资金流分拆 (万元)
    main_force_net    DOUBLE PRECISION DEFAULT 0,      -- 主力净流入
    main_force_ratio  DOUBLE PRECISION DEFAULT 0,      -- 大单净量% (ddejingliang)
    super_large_net   DOUBLE PRECISION DEFAULT 0,      -- 超大单净流入
    large_net         DOUBLE PRECISION DEFAULT 0,      -- 大单净流入
    mid_net           DOUBLE PRECISION DEFAULT 0,      -- 中单净流入
    small_net         DOUBLE PRECISION DEFAULT 0,      -- 小单净流入
    total_net         DOUBLE PRECISION DEFAULT 0,      -- 总净流入
    northbound_net    DOUBLE PRECISION DEFAULT 0,      -- 北向净流入
    -- 评分
    flow_ratio        DOUBLE PRECISION DEFAULT 0,      -- 净流入/市值
    sector_flow       DOUBLE PRECISION DEFAULT 0,      -- 板块资金流向
    flow_score        DOUBLE PRECISION DEFAULT 0,      -- 0-1 归一化
    -- 基本面 (tx_finance)
    pe                DOUBLE PRECISION DEFAULT 0,
    pb                DOUBLE PRECISION DEFAULT 0,
    market_cap        DOUBLE PRECISION DEFAULT 0,      -- 总市值
    -- 元数据
    data_source       VARCHAR(20) DEFAULT '',           -- tx_finance / ths_hot / eastmoney
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_flow_run_id  ON capital_flow_records (run_id);
CREATE INDEX IF NOT EXISTS idx_flow_symbol  ON capital_flow_records (symbol);
CREATE INDEX IF NOT EXISTS idx_flow_date    ON capital_flow_records (created_at DESC);

-- 5. 行业板块资金流 (sector_flow node → 同花顺90行业板块)
--    source: akshare stock_board_industry_summary_ths
-- ============================================================================
CREATE TABLE IF NOT EXISTS sector_flow_records (
    id                  SERIAL PRIMARY KEY,
    run_id              VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    sector_code         VARCHAR(20) DEFAULT '',
    sector_name         VARCHAR(100) NOT NULL,
    change_pct          DOUBLE PRECISION DEFAULT 0,     -- 板块涨跌幅%
    turnover_yi         DOUBLE PRECISION DEFAULT 0,     -- 成交额(亿)
    main_force_net      DOUBLE PRECISION DEFAULT 0,     -- 净流入(亿)
    main_force_ratio    DOUBLE PRECISION DEFAULT 0,     -- 净流入/成交额%
    super_large_net     DOUBLE PRECISION DEFAULT 0,     -- 超大单净流入(亿)
    large_net           DOUBLE PRECISION DEFAULT 0,     -- 大单净流入(亿)
    heat                INT DEFAULT 0,                  -- 活跃度 (up+down count)
    stock_count         INT DEFAULT 0,                  -- 板块股票数
    up_count            INT DEFAULT 0,                  -- 上涨家数
    down_count          INT DEFAULT 0,                  -- 下跌家数
    leading_stock       VARCHAR(10) DEFAULT '',         -- 领涨股代码
    leading_stock_name  VARCHAR(50) DEFAULT '',
    leading_stock_change DOUBLE PRECISION DEFAULT 0,    -- 领涨股涨幅%
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sf_run_id   ON sector_flow_records (run_id);
CREATE INDEX IF NOT EXISTS idx_sf_name     ON sector_flow_records (sector_name);
CREATE INDEX IF NOT EXISTS idx_sf_net      ON sector_flow_records (main_force_net DESC);

-- 6. 龙虎榜 (dragon_tiger_board 产出)
-- ============================================================================
CREATE TABLE IF NOT EXISTS dragon_tiger_records (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    stock_code      VARCHAR(10) NOT NULL,
    stock_name      VARCHAR(50) DEFAULT '',
    trade_date      DATE NOT NULL,
    reason          VARCHAR(200) DEFAULT '',            -- 上榜原因
    buy_seats       JSONB DEFAULT '[]',                -- [{seat, amount}]
    sell_seats      JSONB DEFAULT '[]',
    total_buy       DOUBLE PRECISION DEFAULT 0,        -- 万元
    total_sell      DOUBLE PRECISION DEFAULT 0,
    net_amount      DOUBLE PRECISION DEFAULT 0,        -- 万元
    famous_traders  TEXT[] DEFAULT '{}',               -- 识别到的游资
    trader_signal   VARCHAR(20) DEFAULT '',            -- 合力做多/分歧/出货
    lhb_score       DOUBLE PRECISION DEFAULT 0,        -- 0-1
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_lhb_run_id     ON dragon_tiger_records (run_id);
CREATE INDEX IF NOT EXISTS idx_lhb_stock_code ON dragon_tiger_records (stock_code);
CREATE INDEX IF NOT EXISTS idx_lhb_trade_date ON dragon_tiger_records (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_lhb_traders    ON dragon_tiger_records USING GIN (famous_traders);
CREATE INDEX IF NOT EXISTS idx_lhb_signal     ON dragon_tiger_records (trader_signal);

-- 7. 活跃股票池 (merge_active_stocks 产出, top30)
--    合并 W_FLOW=0.40 + W_CONCEPT=0.35 + W_LHB=0.15
-- ============================================================================
CREATE TABLE IF NOT EXISTS active_stocks (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    symbol          VARCHAR(10) NOT NULL,
    stock_name      VARCHAR(50) DEFAULT '',
    rank            INT DEFAULT 0,
    active_score    DOUBLE PRECISION DEFAULT 0,        -- 综合活跃分 0-1
    flow_score      DOUBLE PRECISION DEFAULT 0,        -- W_FLOW 子分
    concept_score   DOUBLE PRECISION DEFAULT 0,        -- W_CONCEPT 子分
    lhb_score       DOUBLE PRECISION DEFAULT 0,        -- W_LHB 子分
    -- 资金快照
    main_force_net  DOUBLE PRECISION DEFAULT 0,
    ddejingliang    DOUBLE PRECISION DEFAULT 0,        -- 大单净量%
    super_large_net DOUBLE PRECISION DEFAULT 0,
    large_net       DOUBLE PRECISION DEFAULT 0,
    mid_net         DOUBLE PRECISION DEFAULT 0,
    small_net       DOUBLE PRECISION DEFAULT 0,
    amount_wan      DOUBLE PRECISION DEFAULT 0,
    change_pct      DOUBLE PRECISION DEFAULT 0,
    pe              DOUBLE PRECISION DEFAULT 0,
    pb              DOUBLE PRECISION DEFAULT 0,
    market_cap      DOUBLE PRECISION DEFAULT 0,
    data_source     VARCHAR(20) DEFAULT '',
    reasons         TEXT DEFAULT '',                   -- 入选理由
    matched_concepts TEXT[] DEFAULT '{}',              -- 命中的热点概念
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_as_run_id   ON active_stocks (run_id);
CREATE INDEX IF NOT EXISTS idx_as_symbol   ON active_stocks (symbol);
CREATE INDEX IF NOT EXISTS idx_as_rank     ON active_stocks (run_id, rank);

-- 8. 题材概念快照 (sector_flow node → 同花顺热点)
--    每个概念一条, 按天快照
-- ============================================================================
CREATE TABLE IF NOT EXISTS sector_concepts (
    id                  SERIAL PRIMARY KEY,
    run_id              VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    concept_name        VARCHAR(100) NOT NULL,
    concept_id          VARCHAR(20) DEFAULT '',
    leader_stock        VARCHAR(10) DEFAULT '',
    leader_stock_name   VARCHAR(50) DEFAULT '',
    leader_stock_change DOUBLE PRECISION DEFAULT 0,
    change_pct          DOUBLE PRECISION DEFAULT 0,
    heat                INT DEFAULT 0,
    stock_count         INT DEFAULT 0,
    snapshot_date       DATE NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sc_concept ON sector_concepts (concept_name);
CREATE INDEX IF NOT EXISTS idx_sc_date    ON sector_concepts (snapshot_date DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uk_sc_name_date ON sector_concepts (concept_name, snapshot_date);

-- 股票-概念关联 (来自 ingest_event sector tag enrichment)
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_concept_tags (
    id           SERIAL PRIMARY KEY,
    run_id       VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    stock_code   VARCHAR(10) NOT NULL,
    concept_name VARCHAR(100) NOT NULL,
    is_leader    BOOLEAN DEFAULT FALSE,               -- 是否该概念领涨股
    snapshot_date DATE NOT NULL,
    UNIQUE (stock_code, concept_name, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_sct_stock   ON stock_concept_tags (stock_code);
CREATE INDEX IF NOT EXISTS idx_sct_concept ON stock_concept_tags (concept_name);
CREATE INDEX IF NOT EXISTS idx_sct_date    ON stock_concept_tags (snapshot_date DESC);

-- 9. 龙头候选 (generate_candidates 产出) ★ 核心表
-- ============================================================================
CREATE TABLE IF NOT EXISTS leader_candidates (
    id                SERIAL PRIMARY KEY,
    run_id            VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    stock_code        VARCHAR(10) NOT NULL,
    stock_name        VARCHAR(50) DEFAULT '',
    trade_date        DATE NOT NULL,
    rank              INT DEFAULT 0,
    leader_score      DOUBLE PRECISION DEFAULT 0,     -- 龙头概率 0-1
    monster_potential DOUBLE PRECISION DEFAULT 0,     -- 妖股潜力 0-1
    limit_up_prob     DOUBLE PRECISION DEFAULT 0,     -- 涨停概率 0-1
    reasoning         TEXT DEFAULT '',
    sector            VARCHAR(50) DEFAULT '',
    -- 子分数 (调试可解释性)
    sentiment_sub     DOUBLE PRECISION DEFAULT 0,     -- 情绪 0.25
    flow_sub          DOUBLE PRECISION DEFAULT 0,     -- 资金 0.25
    lhb_sub           DOUBLE PRECISION DEFAULT 0,     -- 龙虎榜 0.20
    ml_sub            DOUBLE PRECISION DEFAULT 0,     -- ML集成 0.20
    event_sub         DOUBLE PRECISION DEFAULT 0,     -- 事件记忆 0.10
    sector_tag_sub    DOUBLE PRECISION DEFAULT 0,     -- 题材加分
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_lc_run_id      ON leader_candidates (run_id);
CREATE INDEX IF NOT EXISTS idx_lc_stock_code  ON leader_candidates (stock_code);
CREATE INDEX IF NOT EXISTS idx_lc_trade_date  ON leader_candidates (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_lc_rank        ON leader_candidates (trade_date, rank);
CREATE INDEX IF NOT EXISTS idx_lc_score       ON leader_candidates (trade_date DESC, leader_score DESC);

-- 10. 风险标记 (generate_candidates 产出, 当前空壳)
-- ============================================================================
CREATE TABLE IF NOT EXISTS risk_flags (
    id          SERIAL PRIMARY KEY,
    run_id      VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    stock_code  VARCHAR(10) NOT NULL,
    risk_type   VARCHAR(30) DEFAULT '',               -- 炸板/假突破/退潮/高开低走/接力失败
    severity    DOUBLE PRECISION DEFAULT 0,           -- 0-1
    description TEXT DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rf_run_id     ON risk_flags (run_id);
CREATE INDEX IF NOT EXISTS idx_rf_stock_code ON risk_flags (stock_code);
CREATE INDEX IF NOT EXISTS idx_rf_type       ON risk_flags (risk_type);

-- 11. 激活的历史记忆 (ingest_event → ChromaDB 相似事件召回)
-- ============================================================================
CREATE TABLE IF NOT EXISTS activated_memories (
    id                    SERIAL PRIMARY KEY,
    run_id                VARCHAR(50) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    current_event_title   TEXT DEFAULT '',
    historical_summary    TEXT DEFAULT '',
    similarity            DOUBLE PRECISION DEFAULT 0,  -- cosine 相似度
    lifecycle_stage       VARCHAR(20) DEFAULT '',      -- emerging/active/decaying
    memory_created_at     VARCHAR(30) DEFAULT '',
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_am_run_id ON activated_memories (run_id);

-- ============================================================================
-- 12. 妖股数据湖 (monster stock case base)
-- ============================================================================

CREATE TABLE IF NOT EXISTS monster_stock (
    id                SERIAL PRIMARY KEY,
    stock_code        VARCHAR(10) NOT NULL,
    stock_name        VARCHAR(30) NOT NULL,
    primary_type      VARCHAR(30) NOT NULL,
    secondary_type    VARCHAR(30) NOT NULL,
    tags              TEXT[] DEFAULT '{}',
    sector            VARCHAR(50) DEFAULT '',
    start_date        DATE NOT NULL,
    end_date          DATE NOT NULL,
    trading_days      INT DEFAULT 0,
    start_price       NUMERIC(10,3) DEFAULT 0,
    peak_price        NUMERIC(10,3) DEFAULT 0,
    max_gain_pct      NUMERIC(8,2) DEFAULT 0,
    market_cap_start  NUMERIC(12,2) DEFAULT 0,
    market_cap_peak   NUMERIC(12,2) DEFAULT 0,
    daily_turnover_avg_pre   NUMERIC(6,2) DEFAULT 0,
    daily_turnover_avg_surge NUMERIC(6,2) DEFAULT 0,
    limit_up_count           INT DEFAULT 0,
    consecutive_boards_max   INT DEFAULT 0,
    drawdown_pct      NUMERIC(8,2) DEFAULT 0,
    key_traders       TEXT[] DEFAULT '{}',
    similar_cases     TEXT[] DEFAULT '{}',
    markdown_path     VARCHAR(500) DEFAULT '',
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ms_type   ON monster_stock (primary_type, secondary_type);
CREATE INDEX IF NOT EXISTS idx_ms_code   ON monster_stock (stock_code);
CREATE INDEX IF NOT EXISTS idx_ms_date   ON monster_stock (start_date);
CREATE INDEX IF NOT EXISTS idx_ms_tags   ON monster_stock USING GIN (tags);

-- 妖股日线数据
CREATE TABLE IF NOT EXISTS monster_daily_bar (
    id              SERIAL PRIMARY KEY,
    stock_code      VARCHAR(10) NOT NULL,
    trade_date      DATE NOT NULL,
    open            NUMERIC(10,3) NOT NULL,
    high            NUMERIC(10,3) NOT NULL,
    low             NUMERIC(10,3) NOT NULL,
    close           NUMERIC(10,3) NOT NULL,
    volume          BIGINT DEFAULT 0,
    amount          NUMERIC(16,2) DEFAULT 0,
    turnover_pct    NUMERIC(6,2) DEFAULT 0,
    change_pct      NUMERIC(6,2) DEFAULT 0,
    is_limit_up     BOOLEAN DEFAULT FALSE,
    is_limit_down   BOOLEAN DEFAULT FALSE,
    ma5             NUMERIC(10,3),
    ma10            NUMERIC(10,3),
    ma20            NUMERIC(10,3),
    volume_ratio    NUMERIC(8,2),
    phase           VARCHAR(20) DEFAULT '',
    UNIQUE (stock_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_mdb_code_date ON monster_daily_bar (stock_code, trade_date);
CREATE INDEX IF NOT EXISTS idx_mdb_date      ON monster_daily_bar (trade_date);
CREATE INDEX IF NOT EXISTS idx_mdb_phase     ON monster_daily_bar (stock_code, phase);

-- 妖股分钟线数据
CREATE TABLE IF NOT EXISTS monster_minute_bar (
    id              SERIAL PRIMARY KEY,
    stock_code      VARCHAR(10) NOT NULL,
    trade_date      DATE NOT NULL,
    bar_time        TIME NOT NULL,
    open            NUMERIC(10,3) NOT NULL,
    high            NUMERIC(10,3) NOT NULL,
    low             NUMERIC(10,3) NOT NULL,
    close           NUMERIC(10,3) NOT NULL,
    volume          BIGINT DEFAULT 0,
    amount          NUMERIC(16,2) DEFAULT 0,
    phase           VARCHAR(20) DEFAULT '',
    UNIQUE (stock_code, trade_date, bar_time)
);
CREATE INDEX IF NOT EXISTS idx_mmb_code_date ON monster_minute_bar (stock_code, trade_date);

-- 16. Stock daily bars (OHLCV for backtest)
CREATE TABLE IF NOT EXISTS stock_daily_bars (
    id           SERIAL PRIMARY KEY,
    symbol       VARCHAR(10) NOT NULL,
    trade_date   DATE NOT NULL,
    open         DOUBLE PRECISION DEFAULT 0,
    high         DOUBLE PRECISION DEFAULT 0,
    low          DOUBLE PRECISION DEFAULT 0,
    close        DOUBLE PRECISION DEFAULT 0,
    volume       BIGINT DEFAULT 0,
    amount       DOUBLE PRECISION DEFAULT 0,
    change_pct   DOUBLE PRECISION DEFAULT 0,
    turnover_pct DOUBLE PRECISION DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON stock_daily_bars(symbol);
CREATE INDEX IF NOT EXISTS idx_bars_date ON stock_daily_bars(trade_date);

-- ============================================================================
-- 常用查询视图
-- ============================================================================

-- 某日龙头推荐汇总 (含事件数、报错)
CREATE OR REPLACE VIEW v_daily_leaders AS
SELECT
    lc.trade_date,
    lc.rank,
    lc.stock_code,
    lc.stock_name,
    lc.leader_score,
    lc.monster_potential,
    lc.limit_up_prob,
    lc.reasoning,
    lc.sentiment_sub,
    lc.flow_sub,
    lc.lhb_sub,
    lc.ml_sub,
    lc.event_sub,
    lc.sector_tag_sub,
    pr.event_count,
    pr.errors
FROM leader_candidates lc
JOIN pipeline_runs pr ON lc.run_id = pr.run_id
ORDER BY lc.trade_date DESC, lc.rank ASC;

-- 某股票历史评分走势
CREATE OR REPLACE VIEW v_stock_score_history AS
SELECT
    lc.trade_date,
    lc.stock_code,
    lc.stock_name,
    lc.rank,
    lc.leader_score,
    lc.monster_potential,
    lc.limit_up_prob
FROM leader_candidates lc
ORDER BY lc.stock_code, lc.trade_date DESC;

-- 最近交易日完整面板 (active → candidate → flow 明细)
CREATE OR REPLACE VIEW v_latest_panel AS
SELECT
    a.symbol,
    a.stock_name,
    a.rank          AS active_rank,
    a.active_score,
    a.reasons       AS merge_reasons,
    lc.rank         AS candidate_rank,
    lc.leader_score,
    lc.monster_potential,
    lc.limit_up_prob,
    lc.reasoning    AS candidate_reasoning,
    cf.main_force_net,
    cf.main_force_ratio,
    cf.amount_wan,
    cf.change_pct,
    cf.pe,
    cf.market_cap,
    lhb.trader_signal,
    lhb.famous_traders
FROM active_stocks a
LEFT JOIN leader_candidates lc
    ON a.run_id = lc.run_id AND a.symbol = lc.stock_code
LEFT JOIN capital_flow_records cf
    ON a.run_id = cf.run_id AND a.symbol = cf.symbol
LEFT JOIN dragon_tiger_records lhb
    ON a.run_id = lhb.run_id AND a.symbol = lhb.stock_code
WHERE a.run_id = (
    SELECT run_id FROM pipeline_runs ORDER BY created_at DESC LIMIT 1
)
ORDER BY a.rank ASC;

-- ============================================================================
-- v3 Schema migrations (2026-05-18)
-- FinBERT + LLM audit trail columns
-- ============================================================================
ALTER TABLE events
    ADD COLUMN IF NOT EXISTS llm_prompt   TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS llm_response TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS llm_model    VARCHAR(50) DEFAULT '';

ALTER TABLE sentiment_scores
    ADD COLUMN IF NOT EXISTS finbert_positive DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS finbert_negative DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS finbert_neutral  DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS llm_prompt       TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS llm_response     TEXT DEFAULT '';
