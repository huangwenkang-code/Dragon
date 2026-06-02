// Dragon Engine — SSE streaming types

export interface SSEEvent {
  type: 'stage_update' | 'event_partial' | 'sentiment_partial' | 'candidate_partial' | 'error' | 'done'
  payload: unknown
  timestamp: string
}

export interface AIThinkingState {
  active_stage: string
  message: string
  progress: number       // 0-100
  reasoning: string      // LLM 推理原文片段
  arrived_at: string
}

export interface StreamConnection {
  status: 'disconnected' | 'connecting' | 'connected' | 'error'
  last_event_at?: string
  error_message?: string
}
