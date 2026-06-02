// GET /events/:runId
import client from './client'
import type { PipelineEvent } from '@/types/api'

export function fetchEvents(runId: string) {
  return client.get<PipelineEvent[]>(`/events/${runId}`)
}
