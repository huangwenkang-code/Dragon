// GET /history — list all previous pipeline runs
import client from './client'
import type { PipelineRun } from '@/types/pipeline'

export function fetchHistory() {
  return client.get<PipelineRun[]>('/history')
}
