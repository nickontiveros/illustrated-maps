import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useLandmarks(projectName: string | undefined) {
  return useQuery({
    queryKey: ['project', projectName, 'landmarks'],
    queryFn: () => api.listLandmarks(projectName!),
    enabled: !!projectName,
  });
}

export function useCreateLandmark(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      name: string;
      latitude: number;
      longitude: number;
      photo?: string;
      logo?: string;
      scale?: number;
      z_index?: number;
      rotation?: number;
    }) => api.createLandmark(projectName, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectName] });
    },
  });
}

export function useUpdateLandmark(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      landmarkName,
      data,
    }: {
      landmarkName: string;
      data: {
        latitude?: number;
        longitude?: number;
        scale?: number;
        z_index?: number;
        rotation?: number;
      };
    }) => api.updateLandmark(projectName, landmarkName, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
    },
  });
}

export function useDeleteLandmark(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (landmarkName: string) => api.deleteLandmark(projectName, landmarkName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectName] });
    },
  });
}

export function useIllustrateLandmark(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (landmarkName: string) => api.illustrateLandmark(projectName, landmarkName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
    },
  });
}

export function useUploadLandmarkPhoto(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ landmarkName, file }: { landmarkName: string; file: File }) =>
      api.uploadLandmarkPhoto(projectName, landmarkName, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
    },
  });
}

export function useDiscoverLandmarks(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data?: { min_importance_score?: number; max_landmarks?: number }) =>
      api.discoverLandmarks(projectName, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
    },
  });
}

export function useIllustrateAllLandmarks(projectName: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.illustrateAllLandmarks(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName, 'landmarks'] });
    },
  });
}
