import { useQuery } from '@tanstack/react-query'

import { api, type ModelsResponse } from '@/lib/api'

/** Loads the model/tool registry so the header can report real tool availability. */
export function useModels() {
  return useQuery<ModelsResponse>({
    queryKey: ['models'],
    queryFn: api.getModels,
    staleTime: 60_000,
    retry: 1,
  })
}
