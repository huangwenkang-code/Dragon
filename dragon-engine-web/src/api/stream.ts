// SSE EventSource wrapper — real-time pipeline streaming
import type { SSEEvent } from '@/types/stream'

export function createPipelineStream(
  runId: string,
  onEvent: (ev: SSEEvent) => void,
  onError?: (err: Event) => void,
): EventSource {
  const es = new EventSource(`http://localhost:8000/stream/${runId}`)

  es.addEventListener('stage_update', (e: MessageEvent) => {
    onEvent({ type: 'stage_update', payload: JSON.parse(e.data), timestamp: new Date().toISOString() })
  })
  es.addEventListener('event_partial', (e: MessageEvent) => {
    onEvent({ type: 'event_partial', payload: JSON.parse(e.data), timestamp: new Date().toISOString() })
  })
  es.addEventListener('sentiment_partial', (e: MessageEvent) => {
    onEvent({ type: 'sentiment_partial', payload: JSON.parse(e.data), timestamp: new Date().toISOString() })
  })
  es.addEventListener('candidate_partial', (e: MessageEvent) => {
    onEvent({ type: 'candidate_partial', payload: JSON.parse(e.data), timestamp: new Date().toISOString() })
  })
  es.addEventListener('done', (e: MessageEvent) => {
    onEvent({ type: 'done', payload: JSON.parse(e.data), timestamp: new Date().toISOString() })
    es.close()
  })
  es.addEventListener('error', (e: MessageEvent) => {
    onEvent({ type: 'error', payload: e.data, timestamp: new Date().toISOString() })
  })

  es.onerror = (err) => {
    onError?.(err)
  }

  return es
}
