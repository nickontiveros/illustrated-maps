import { useEffect, useRef, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useAppStore } from '@/stores/appStore';
import type { WSMessage } from '@/types';

/**
 * Global hook that polls for active tasks and manages WebSocket connections.
 * Should be mounted once at the app root level so it persists across navigation.
 */
export function useGlobalGeneration() {
  const { activeGenerations, setActiveGeneration, updateGenerationProgress } = useAppStore();
  const wsRefs = useRef<Record<string, WebSocket>>({});
  const queryClient = useQueryClient();

  // Poll active tasks every 3 seconds
  const { data: activeTasks } = useQuery({
    queryKey: ['activeTasks'],
    queryFn: () => api.getActiveTasks(),
    refetchInterval: 3000,
  });

  // Sync discovered active tasks into Zustand store
  useEffect(() => {
    if (!activeTasks) return;

    const activeProjectNames = new Set(activeTasks.map((t) => t.project_name));

    // Add newly discovered tasks
    for (const task of activeTasks) {
      const existing = activeGenerations[task.project_name];
      if (!existing || existing.taskId !== task.task_id) {
        setActiveGeneration(task.project_name, {
          taskId: task.task_id,
          projectName: task.project_name,
          progress: task.progress
            ? {
                status: 'running',
                total_tiles: task.progress.total_tiles,
                completed_tiles: task.progress.completed_tiles,
                failed_tiles: task.progress.failed_tiles,
                current_tile: null,
                elapsed_seconds: task.progress.elapsed_seconds,
                estimated_remaining_seconds: task.progress.estimated_remaining_seconds,
                error: null,
              }
            : null,
        });
      } else if (task.progress && existing) {
        // Update progress from poll (WebSocket is more real-time but this is the fallback)
        updateGenerationProgress(task.project_name, {
          status: 'running',
          total_tiles: task.progress.total_tiles,
          completed_tiles: task.progress.completed_tiles,
          failed_tiles: task.progress.failed_tiles,
          current_tile: null,
          elapsed_seconds: task.progress.elapsed_seconds,
          estimated_remaining_seconds: task.progress.estimated_remaining_seconds,
          error: null,
        });
      }
    }

    // Remove generations that are no longer active on the server
    for (const projectName of Object.keys(activeGenerations)) {
      if (!activeProjectNames.has(projectName)) {
        setActiveGeneration(projectName, null);
        // Invalidate project-related queries so UI refreshes
        queryClient.invalidateQueries({ queryKey: ['project', projectName] });
        queryClient.invalidateQueries({ queryKey: ['projects'] });
      }
    }
  }, [activeTasks]);

  // Manage WebSocket connections for each active generation
  const connectWs = useCallback(
    (projectName: string, taskId: string) => {
      // Don't double-connect
      if (wsRefs.current[projectName]?.readyState === WebSocket.OPEN) return;

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/ws/generation/${taskId}`;
      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          if (message.type === 'progress') {
            updateGenerationProgress(projectName, message.data);
          } else if (message.type === 'done') {
            setActiveGeneration(projectName, null);
            queryClient.invalidateQueries({ queryKey: ['project', projectName] });
            queryClient.invalidateQueries({ queryKey: ['projects'] });
            queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        delete wsRefs.current[projectName];
      };

      wsRefs.current[projectName] = ws;
    },
    [updateGenerationProgress, setActiveGeneration, queryClient]
  );

  // Connect WebSockets for active generations
  useEffect(() => {
    for (const [projectName, gen] of Object.entries(activeGenerations)) {
      connectWs(projectName, gen.taskId);
    }

    // Clean up WebSockets for removed generations
    for (const projectName of Object.keys(wsRefs.current)) {
      if (!activeGenerations[projectName]) {
        wsRefs.current[projectName]?.close();
        delete wsRefs.current[projectName];
      }
    }
  }, [activeGenerations, connectWs]);

  // Cleanup all WebSockets on unmount
  useEffect(() => {
    return () => {
      for (const ws of Object.values(wsRefs.current)) {
        ws.close();
      }
      wsRefs.current = {};
    };
  }, []);

  return { activeGenerations, activeTasks };
}
