import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useTileGrid(projectName: string | undefined) {
  return useQuery({
    queryKey: ['project', projectName, 'tiles'],
    queryFn: () => api.getTileGrid(projectName!),
    enabled: !!projectName,
  });
}

export function useTileInfo(projectName: string | undefined, col: number, row: number) {
  return useQuery({
    queryKey: ['project', projectName, 'tile', col, row],
    queryFn: () => api.getTileInfo(projectName!, col, row),
    enabled: !!projectName,
  });
}

export function useRegenerateTile(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ col, row, force = false }: { col: number; row: number; force?: boolean }) =>
      api.regenerateTile(projectName, col, row, force),
    onSuccess: (_, { col, row }) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'tiles'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'tile', col, row] });
    },
  });
}

export function useGenerationStatus(projectName: string | undefined) {
  return useQuery({
    queryKey: ['project', projectName, 'generation', 'status'],
    queryFn: () => api.getGenerationStatus(projectName!),
    enabled: !!projectName,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === 'running') {
        return 2000; // Poll every 2 seconds while running
      }
      return false;
    },
  });
}

export function useStartGeneration(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (options?: { skip_existing?: boolean; tile_filter?: [number, number][] }) =>
      api.startGeneration(projectName, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'generation'] });
    },
  });
}

export function useCancelGeneration(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.cancelGeneration(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'generation'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'tiles'] });
    },
  });
}
