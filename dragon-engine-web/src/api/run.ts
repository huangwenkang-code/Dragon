// POST /run — execute full pipeline
import client from './client'
import type { RunRequest, RunResponse } from '@/types/api'

export function runPipeline(req: RunRequest) {
  return client.post<RunResponse>('/run', req)
}
