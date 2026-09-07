import { useCallback, useState } from 'react'

import { useQuery, useQueryClient } from '@tanstack/react-query'

import { ApiError, type AnalysisResult, type HistoryResponse, api } from '@/lib/api'

/** Recent analyses, refreshed whenever a new one completes. */
export function useHistory() {
  return useQuery<HistoryResponse>({
    queryKey: ['history'],
    queryFn: api.getHistory,
    staleTime: 5_000,
  })
}

/**
 * Runs an analysis and holds the result.
 *
 * The backend returns a result envelope even when the outcome is `rejected` or
 * `unavailable`, so those are stored as results (with their reasons and trace) rather
 * than thrown away as errors. Only transport/validation failures become `error`.
 */
export function useAnalysis() {
  const queryClient = useQueryClient()
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const analyze = useCallback(
    async (query: string, assetIds: string[]) => {
      setIsRunning(true)
      setError(null)
      try {
        const outcome = await api.analyze(query, assetIds)
        setResult(outcome)
        void queryClient.invalidateQueries({ queryKey: ['history'] })
        return outcome
      } catch (caught) {
        setError(
          caught instanceof ApiError
            ? caught.message
            : 'The analysis request could not be sent.',
        )
        return null
      } finally {
        setIsRunning(false)
      }
    },
    [queryClient],
  )

  const load = useCallback(async (id: string) => {
    setIsRunning(true)
    setError(null)
    try {
      setResult(await api.getAnalysis(id))
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load that analysis.')
    } finally {
      setIsRunning(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { result, isRunning, error, analyze, load, reset }
}
