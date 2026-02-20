import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { BoundingBox, OutputSettings, StyleSettings, TileSettings, BorderSettings, NarrativeSettings } from '@/types';

export function useProjects() {
  return useQuery({
    queryKey: ['projects'],
    queryFn: () => api.listProjects(),
  });
}

export function useProject(name: string | undefined) {
  return useQuery({
    queryKey: ['project', name],
    queryFn: () => api.getProject(name!),
    enabled: !!name,
  });
}

export function useProjectCost(name: string | undefined) {
  return useQuery({
    queryKey: ['project', name, 'cost'],
    queryFn: () => api.getProjectCostEstimate(name!),
    enabled: !!name,
  });
}

export function useCreateProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      name: string;
      region: BoundingBox;
      output?: OutputSettings;
      style?: StyleSettings;
      tiles?: TileSettings;
    }) => api.createProject(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}

export function useUpdateProject(name: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      output?: OutputSettings;
      style?: StyleSettings;
      tiles?: TileSettings;
      title?: string;
      subtitle?: string;
      border?: BorderSettings;
      narrative?: NarrativeSettings;
    }) => api.updateProject(name, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', name] });
    },
  });
}

export function useDeleteProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, deleteCache = true }: { name: string; deleteCache?: boolean }) =>
      api.deleteProject(name, deleteCache),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}
