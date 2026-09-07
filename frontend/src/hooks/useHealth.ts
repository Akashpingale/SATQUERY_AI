import { useQuery } from '@tanstack/react-query'

import { api, type HealthResponse } from '@/lib/api'

/** Polls backend health so the header can show live system status. */
export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: api.getHealth,
    refetchInterval: 30_000,
    retry: 1,
    staleTime: 10_000,
  })
}
