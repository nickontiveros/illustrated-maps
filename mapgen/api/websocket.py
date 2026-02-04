"""WebSocket handlers for real-time updates."""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .schemas import GenerationProgress, GenerationStatus
from .tasks import task_manager

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # task_id -> list of connected websockets
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        """Accept a new WebSocket connection for a task."""
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str):
        """Remove a WebSocket connection."""
        if task_id in self.active_connections:
            try:
                self.active_connections[task_id].remove(websocket)
            except ValueError:
                pass
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def broadcast(self, task_id: str, message: dict):
        """Broadcast a message to all connections for a task."""
        if task_id not in self.active_connections:
            return

        dead_connections = []
        for websocket in self.active_connections[task_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.append(websocket)

        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws, task_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/generation/{task_id}")
async def generation_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for generation progress updates.

    Connect to receive real-time updates for a generation task.

    Message format:
    {
        "type": "progress" | "tile_complete" | "error" | "done",
        "data": {
            "status": "running" | "completed" | "failed" | "cancelled",
            "total_tiles": int,
            "completed_tiles": int,
            "failed_tiles": int,
            "current_tile": [col, row] | null,
            "elapsed_seconds": float,
            "estimated_remaining_seconds": float | null,
            "error": string | null
        }
    }
    """
    # Verify task exists
    task_info = await task_manager.get_task(task_id)
    if task_info is None:
        await websocket.close(code=4004, reason="Task not found")
        return

    await manager.connect(websocket, task_id)

    # Send initial status
    if task_info.progress:
        await websocket.send_json({
            "type": "progress",
            "data": _progress_to_dict(task_info.progress),
        })
    else:
        await websocket.send_json({
            "type": "status",
            "data": {"status": task_info.status.value},
        })

    # Register callback for progress updates
    async def on_progress(progress: GenerationProgress):
        await manager.broadcast(task_id, {
            "type": "progress",
            "data": _progress_to_dict(progress),
        })

    task_manager.register_websocket(task_id, on_progress)

    try:
        while True:
            # Wait for client messages (ping/pong, etc.)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client messages if needed
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

            # Check if task is done
            task_info = await task_manager.get_task(task_id)
            if task_info and task_info.status in [
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.CANCELLED,
            ]:
                await websocket.send_json({
                    "type": "done",
                    "data": {
                        "status": task_info.status.value,
                        "error": task_info.error,
                    },
                })
                break

    except WebSocketDisconnect:
        pass
    finally:
        task_manager.unregister_websocket(task_id, on_progress)
        manager.disconnect(websocket, task_id)


def _progress_to_dict(progress) -> dict:
    """Convert GenerationProgress to dict for JSON serialization."""
    return {
        "status": "running",
        "total_tiles": progress.total_tiles,
        "completed_tiles": progress.completed_tiles,
        "failed_tiles": progress.failed_tiles,
        "current_tile": list(progress.current_tile) if progress.current_tile else None,
        "elapsed_seconds": progress.elapsed_time,
        "estimated_remaining_seconds": progress.estimated_remaining if progress.completed_tiles > 0 else None,
    }


@router.websocket("/ws/project/{project_name}")
async def project_websocket(websocket: WebSocket, project_name: str):
    """WebSocket endpoint for general project updates.

    Receives updates about any changes to the project (tile changes, etc.)
    """
    await websocket.accept()

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        pass
