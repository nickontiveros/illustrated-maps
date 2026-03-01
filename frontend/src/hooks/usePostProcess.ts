import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { PostProcessStatus } from '@/types';

export function usePostProcessStatus(projectName: string | undefined) {
  return useQuery<PostProcessStatus>({
    queryKey: ['postprocess-status', projectName],
    queryFn: () => api.getPostProcessStatus(projectName!),
    enabled: !!projectName,
    refetchInterval: 5000,
  });
}

export function useComposeLandmarks(projectName: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.composeLandmarks(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['postprocess-status', projectName] });
    },
  });
}

export function useAddLabels(projectName: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (includeShields: boolean = true) => api.addLabels(projectName, includeShields),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['postprocess-status', projectName] });
    },
  });
}

export function useApplyPerspective(projectName: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params?: {
      angle?: number;
      convergence?: number;
      vertical_scale?: number;
      horizon_margin?: number;
    }) => api.applyPerspective(projectName, params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['postprocess-status', projectName] });
    },
  });
}

export function useAddBorder(projectName: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.addBorder(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['postprocess-status', projectName] });
    },
  });
}

export function useStartOutpaint(projectName: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.startOutpaint(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['postprocess-status', projectName] });
    },
  });
}

export function useStartPipeline(projectName: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ steps, includeShields = true }: { steps: string[]; includeShields?: boolean }) =>
      api.startPipeline(projectName, steps, includeShields),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['postprocess-status', projectName] });
    },
  });
}
