"""
Task Queue System
=================
비동기 태스크 관리 및 상태 추적.
"""

from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import List, Optional
import uuid

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class CoworkTask:
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    agent_type: str = "brain"  # brain, reasoner, tool, rag, etc.
    dependencies: List[str] = field(default_factory=list) # IDs of tasks that must finish first

class TaskQueue:
    """단순 FIFO 태스크 큐 (In-Memory)"""
    
    def __init__(self):
        self._queue = [] # Simple list for inspection, though processing is sequential
        
    def add_task(self, description: str, agent_type: str = "brain") -> CoworkTask:
        task = CoworkTask(description=description, agent_type=agent_type)
        self._queue.append(task)
        return task
        
    def get_pending_tasks(self) -> List[CoworkTask]:
        return [t for t in self._queue if t.status == TaskStatus.PENDING]
        
    def get_all_tasks(self) -> List[CoworkTask]:
        return self._queue
        
    def mark_completed(self, task_id: str, result: str):
        for t in self._queue:
            if t.id == task_id:
                t.status = TaskStatus.COMPLETED
                t.result = result
                break
                
    def mark_failed(self, task_id: str, error: str):
         for t in self._queue:
            if t.id == task_id:
                t.status = TaskStatus.FAILED
                t.result = error
                break
