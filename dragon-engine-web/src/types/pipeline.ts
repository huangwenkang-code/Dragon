// Dragon Engine — pipeline & DAG types

export type PipelineStageName =
  | 'capital_flow'
  | 'ths_hot'
  | 'dragon_tiger_board'
  | 'merge_active_stocks'
  | 'find_news_double_layer'
  | 'analyze_sentiment'
  | 'generate_candidates'

export interface PipelineStage {
  name: PipelineStageName
  label: string
  status: 'pending' | 'running' | 'completed' | 'error'
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface DAGNode {
  id: string
  label: string
  stage: PipelineStageName
  status: 'pending' | 'active' | 'done' | 'blocked'
  x: number
  y: number
  metadata: Record<string, unknown>
}

export interface DAGEdge {
  from: string
  to: string
  label?: string
}

export interface PipelineRun {
  run_id: string
  trade_date: string
  status: 'running' | 'completed' | 'failed'
  stages: PipelineStage[]
  dag: {
    nodes: DAGNode[]
    edges: DAGEdge[]
  }
  started_at: string
  completed_at?: string
}
