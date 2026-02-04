import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSMessage, GenerationProgress } from '@/types';

interface UseWebSocketOptions {
  onProgress?: (progress: GenerationProgress) => void;
  onTileComplete?: (col: number, row: number) => void;
  onError?: (error: string) => void;
  onDone?: (status: string, error: string | null) => void;
  reconnect?: boolean;
}

export function useGenerationWebSocket(taskId: string | null, options: UseWebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState(false);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (!taskId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/generation/${taskId}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'progress':
            setProgress(message.data);
            options.onProgress?.(message.data);
            break;
          case 'tile_complete':
            options.onTileComplete?.(message.data.col, message.data.row);
            break;
          case 'error':
            options.onError?.(message.data.error);
            break;
          case 'done':
            options.onDone?.(message.data.status, message.data.error);
            break;
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;

      // Reconnect if enabled and not intentionally closed
      if (options.reconnect && taskId) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 2000);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, [taskId, options]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (taskId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [taskId, connect, disconnect]);

  return {
    isConnected,
    progress,
    disconnect,
    reconnect: connect,
  };
}
