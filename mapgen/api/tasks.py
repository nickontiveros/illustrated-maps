"""Background task management for long-running operations."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..models.project import Project
from ..services.generation_service import GenerationProgress, GenerationService
from .schemas import GenerationStatus


@dataclass
class TaskInfo:
    """Information about a running task."""

    task_id: str
    project_name: str
    task_type: str
    status: GenerationStatus
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[GenerationProgress] = None
    error: Optional[str] = None
    result: Any = None

    # Internal
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class TaskManager:
    """Manages background tasks for generation, repair, etc."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._websocket_clients: dict[str, list[Callable]] = {}  # task_id -> callbacks
        self._lock = asyncio.Lock()

    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return str(uuid.uuid4())[:8]

    async def create_task(
        self,
        project_name: str,
        task_type: str,
        coro: asyncio.coroutine,
    ) -> TaskInfo:
        """Create and start a new background task.

        Args:
            project_name: Name of the project
            task_type: Type of task (generation, repair, etc.)
            coro: Coroutine to run

        Returns:
            TaskInfo for the created task
        """
        task_id = self.generate_task_id()

        task_info = TaskInfo(
            task_id=task_id,
            project_name=project_name,
            task_type=task_type,
            status=GenerationStatus.IDLE,
        )

        async def wrapper():
            try:
                task_info.status = GenerationStatus.RUNNING
                task_info.started_at = datetime.now()
                result = await coro
                task_info.result = result
                task_info.status = GenerationStatus.COMPLETED
            except asyncio.CancelledError:
                task_info.status = GenerationStatus.CANCELLED
            except Exception as e:
                task_info.status = GenerationStatus.FAILED
                task_info.error = str(e)
            finally:
                task_info.completed_at = datetime.now()

        async with self._lock:
            task_info._task = asyncio.create_task(wrapper())
            self._tasks[task_id] = task_info

        return task_info

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID."""
        return self._tasks.get(task_id)

    async def get_project_tasks(self, project_name: str) -> list[TaskInfo]:
        """Get all tasks for a project."""
        return [t for t in self._tasks.values() if t.project_name == project_name]

    async def get_active_task(self, project_name: str, task_type: str) -> Optional[TaskInfo]:
        """Get the active task of a specific type for a project."""
        for task in self._tasks.values():
            if (
                task.project_name == project_name
                and task.task_type == task_type
                and task.status == GenerationStatus.RUNNING
            ):
                return task
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if task was cancelled, False if not found or not running
        """
        task_info = self._tasks.get(task_id)
        if task_info is None:
            return False

        if task_info.status != GenerationStatus.RUNNING:
            return False

        task_info._cancel_event.set()
        if task_info._task:
            task_info._task.cancel()

        return True

    async def cancel_all(self):
        """Cancel all running tasks."""
        for task_id in list(self._tasks.keys()):
            await self.cancel_task(task_id)

    async def update_progress(self, task_id: str, progress: GenerationProgress):
        """Update task progress and notify WebSocket clients.

        Args:
            task_id: Task ID
            progress: Updated progress
        """
        task_info = self._tasks.get(task_id)
        if task_info:
            task_info.progress = progress

            # Notify WebSocket clients
            callbacks = self._websocket_clients.get(task_id, [])
            for callback in callbacks:
                try:
                    await callback(progress)
                except Exception:
                    pass  # Ignore callback errors

    def register_websocket(self, task_id: str, callback: Callable):
        """Register a WebSocket callback for task updates.

        Args:
            task_id: Task ID to subscribe to
            callback: Async callback function
        """
        if task_id not in self._websocket_clients:
            self._websocket_clients[task_id] = []
        self._websocket_clients[task_id].append(callback)

    def unregister_websocket(self, task_id: str, callback: Callable):
        """Unregister a WebSocket callback.

        Args:
            task_id: Task ID
            callback: Callback to remove
        """
        if task_id in self._websocket_clients:
            try:
                self._websocket_clients[task_id].remove(callback)
            except ValueError:
                pass

    async def cleanup_completed(self, max_age_seconds: int = 3600):
        """Remove completed tasks older than max_age_seconds."""
        now = datetime.now()
        to_remove = []

        for task_id, task_info in self._tasks.items():
            if task_info.completed_at:
                age = (now - task_info.completed_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(task_id)

        async with self._lock:
            for task_id in to_remove:
                del self._tasks[task_id]
                self._websocket_clients.pop(task_id, None)


# Global task manager instance
task_manager = TaskManager()
