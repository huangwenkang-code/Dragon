// Dragon Engine — API request/response types

export interface RunRequest {
  trade_date: string
  top_n: number
  force?: boolean
}

export interface CapitalFlowRecord {
  symbol: string
  stock_name: string
  price: number
  change_pct: number
  main_force_net: number      // 主力净流入(万元)
  main_force_ratio: number    // 主力净流入占比(%)
  super_large_net: number     // 超大单净流入
  large_net: number           // 大单净流入
  mid_net: number             // 中单净流入
  small_net: number           // 小单净流入
  northbound_net: number      // 北向净流入(万元)
  total_net: number           // 总净流入(万元)
  flow_ratio: number          // 净流入占比
  sector_flow: number         // 板块资金流向
  flow_score: number          // 资金流入评分 0-1
  pe: number                  // PE
  pb: number                  // PB
  market_cap: number          // 总市值(亿)
  _source: string             // 数据来源
}

export interface SectorFlowRecord {
  sector_code: string
  sector_name: string
  change_pct: number
  main_force_net: number
  main_force_ratio: number
  super_large_net: number
  large_net: number
  leading_stock: string
  leading_stock_name: string
  leading_stock_change: number
}

export interface CapitalFlowSummary {
  total_main_inflow: number
  top_sectors: string[]
  scanned_stocks: number
  scan_time: string
}

export interface ActiveStock {
  symbol: string
  stock_name: string
  active_score: number
  flow_score: number
  concept_score: number
  lhb_score: number
  main_force_net: number
  change_pct: number
  pe: number
  pb: number
  market_cap: number
  reasons: string
  matched_concepts: string[]
  rank: number
}

export interface DragonTigerRecord {
  stock_code: string
  stock_name: string
  trade_date: string
  reason: string
  buy_seats: SeatRecord[]
  sell_seats: SeatRecord[]
  total_buy: number
  total_sell: number
  net_amount: number
  famous_traders: string[]
  trader_signal: string
  lhb_score: number
}

export interface SeatRecord {
  seat: string
  amount: number
}

export interface ActivatedMemory {
  current_event_title: string
  historical_summary: string
  similarity: number
  lifecycle_stage: string
  created_at: string
}

export interface RunResponse {
  run_id: string
  status: string
  events: PipelineEvent[]
  sentiment_scores: SentimentScore[]
  capital_flow_records: CapitalFlowRecord[]
  sector_flow_records: SectorFlowRecord[]
  capital_flow_summary: CapitalFlowSummary
  active_stocks: ActiveStock[]
  dragon_tiger_records: DragonTigerRecord[]
  leader_candidates: LeaderCandidate[]
  risk_flags: RiskFlag[]
  metadata: Record<string, unknown>
}

export interface PipelineEvent {
  id: string
  event_id: string
  event_type: string
  title: string
  content: string
  summary: string
  narrative: string
  source: string
  publish_time: string
  symbol_list: string[]
  sector_list: string[]
  sector_tags: string[]
  event_strength: number
  heat_score: number
  strength: number
  novelty: number
  scope: string
  keywords: string[]
  timestamp: string
  llm_prompt?: string
  llm_response?: string
  llm_model?: string
}

export interface SentimentScore {
  target_id: string
  target_type: string
  symbol: string
  sentiment_score: number
  narrative_score: number
  hype_score: number
  consistency_score: number
  risk_score: number
  confidence: number
  heat: number
  consensus: number
  diffusion_speed: number
  narrative_strength: number
  keywords: string[]
  timestamp: string
  finbert_positive?: number
  finbert_negative?: number
  finbert_neutral?: number
  llm_prompt?: string
  llm_response?: string
}

export interface MonsterMatchItem {
  stock_code: string
  stock_name: string
  similarity: number
  primary_type: string
  max_gain_pct: number
  trading_days: number
  match_reasons: string[]
}

export interface MonsterReference {
  top_matches: MonsterMatchItem[]
  summary: string
}

export interface LeaderCandidate {
  stock_code: string
  stock_name: string
  leader_score: number
  monster_potential: number
  limit_up_prob: number
  reasoning: string
  sector: string
  rank: number
  monster_reference?: MonsterReference
}

export interface RiskFlag {
  stock_code: string
  risk_type: string
  severity: number
  description: string
  timestamp: string
}

// ---------------------------------------------------------------------------
// Backtest & Token types
// ---------------------------------------------------------------------------

export interface BacktestTrade {
  stock_code: string
  stock_name: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  shares: number
  cost: number
  proceeds: number
  pnl: number
  pnl_pct: number
  entry_commission: number
  exit_commission: number
  stamp_duty: number
  net_pnl: number
  entry_score: number
  exit_score: number
  exit_reason: string
  holding_days: number
  cash_after_trade?: number  // available cash after this trade closed
}

export interface BacktestResult {
  strategy_name: string
  start_date: string
  end_date: string
  initial_capital: number
  final_equity: number
  total_return_pct: number
  max_drawdown_pct: number
  sharpe_ratio: number
  win_rate: number
  total_trades: number
  total_commission: number
  total_stamp_duty: number
  trades: BacktestTrade[]
  daily_snapshots: { date: string; equity: number; cash: number }[]
}

export interface StrategyConfig {
  name: string
  description: string
  entry_rules: { type: string; params: Record<string, any> }[]
  exit_rules: { type: string; params: Record<string, any> }[]
  allocator: { type: string; params: Record<string, any> }
  max_positions: number
  max_position_pct: number
  initial_capital: number
  daily_cash_pct: number
  commission_rate?: number
  stamp_duty_rate?: number
  min_commission?: number
  gap_up_pct?: number | null
  enable_limit_up_filter?: boolean
  is_system?: boolean
}

export interface TokenUsageRow {
  run_id: string
  trade_date: string
  token_usage: {
    total_prompt_tokens: number
    total_completion_tokens: number
    total_tokens: number
    total_cost: number
    records: any[]
  }
}
