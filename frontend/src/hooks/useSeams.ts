import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useSeams(projectName: string | undefined) {
  return useQuery({
    queryKey: ['project', projectName, 'seams'],
    queryFn: () => api.listSeams(projectName!),
    enabled: !!projectName,
  });
}

export function useSeam(projectName: string | undefined, seamId: string | undefined) {
  return useQuery({
    queryKey: ['project', projectName, 'seam', seamId],
    queryFn: () => api.getSeam(projectName!, seamId!),
    enabled: !!projectName && !!seamId,
  });
}

export function useRepairSeam(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (seamId: string) => api.repairSeam(projectName, seamId),
    onSuccess: (_, seamId) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'seams'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'seam', seamId] });
    },
  });
}

export function useRepairSeamsBatch(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (seamIds: string[]) => api.repairSeamsBatch(projectName, seamIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'seams'] });
    },
  });
}

export function useRepairAllSeams(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.repairAllSeams(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'seams'] });
    },
  });
}
