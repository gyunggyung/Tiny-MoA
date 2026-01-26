from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

class BaseWorker(ABC):
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger

    @abstractmethod
    def execute(self, task_description: str, **kwargs) -> str:
        """Execute the given task and return the result as a string."""
        pass
