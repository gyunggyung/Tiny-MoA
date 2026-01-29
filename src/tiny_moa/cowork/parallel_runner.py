"""
Parallel Task Runner
====================
병렬 태스크 실행 및 종속성 관리.
"""

import concurrent.futures
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass
import threading

@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None

class ParallelRunner:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.results = {}
        self.lock = threading.Lock()

    def run_tasks(self, tasks: List[Dict[str, Any]], execute_fn: Callable) -> Dict[str, TaskResult]:
        """
        tasks: [{"id": "...", "description": "...", "agent": "..."}]
        execute_fn: (task_dict) -> result_string
        """
        future_to_task = {
            self.executor.submit(execute_fn, task): task for task in tasks
        }
        
        final_results = {}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result(timeout=60)
                final_results[task['id']] = TaskResult(
                    task_id=task['id'],
                    success=True,
                    result=data
                )
            except concurrent.futures.TimeoutError:
                 final_results[task['id']] = TaskResult(
                    task_id=task['id'],
                    success=False,
                    result=None,
                    error="Task timed out after 60s"
                )
            except Exception as exc:
                final_results[task['id']] = TaskResult(
                    task_id=task['id'],
                    success=False,
                    result=None,
                    error=str(exc)
                )
        return final_results
