import client from './client'

export interface BackfillBarsResult {
  status: string
  symbols: number
  rows: number
  failed: number
  message?: string
}

/** POST /backfill/bars — fill missing OHLCV bars via mootdx */
export function backfillBars(days: number = 7, allStocks: boolean = false) {
  return client.post<BackfillBarsResult>('/backfill/bars', null, {
    params: { days, all_stocks: allStocks },
  })
}
