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

export function useTileOffset(projectName: string | undefined, col: number, row: number) {
  return useQuery({
    queryKey: ['project', projectName, 'tile', col, row, 'offset'],
    queryFn: () => api.getTileOffset(projectName!, col, row),
    enabled: !!projectName,
  });
}

export function useSetTileOffset(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ col, row, dx, dy }: { col: number; row: number; dx: number; dy: number }) =>
      api.setTileOffset(projectName, col, row, dx, dy),
    onSuccess: (_, { col, row }) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'tile', col, row, 'offset'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'tiles'] });
    },
  });
}

export function useStyleReference(projectName: string | undefined) {
  return useQuery({
    queryKey: ['project', projectName, 'style-reference'],
    queryFn: () => api.hasStyleReference(projectName!),
    enabled: !!projectName,
  });
}

export function useUploadStyleReference(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => api.uploadStyleReference(projectName, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'style-reference'] });
    },
  });
}

export function useDeleteStyleReference(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.deleteStyleReference(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'style-reference'] });
    },
  });
}
