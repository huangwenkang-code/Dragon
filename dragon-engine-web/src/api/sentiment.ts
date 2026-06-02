// GET /sentiment/:runId
import client from './client'
import type { SentimentScore } from '@/types/api'

export function fetchSentiment(runId: string) {
  return client.get<SentimentScore[]>(`/sentiment/${runId}`)
}
