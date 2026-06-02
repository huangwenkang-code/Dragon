// POST /replay/:runId
import client from './client'
import type { RunResponse } from '@/types/api'

export function replayRun(runId: string) {
  return client.post<RunResponse>(`/replay/${runId}`)
}
