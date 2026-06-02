// GET /leaders/:runId
import client from './client'
import type { LeaderCandidate } from '@/types/api'

export function fetchLeaders(runId: string) {
  return client.get<LeaderCandidate[]>(`/leaders/${runId}`)
}
