import re
from pathlib import Path
from src.tiny_moa.cowork.workers.base import BaseWorker

class ResearchWorker(BaseWorker):
    def __init__(self, name: str, logger, orchestrator):
        super().__init__(name, logger)
        self.orchestrator = orchestrator

    def execute(self, task_description: str, **kwargs) -> str:
        self.logger.info(f"[{self.name}] Starting research: {task_description}")
        
        # Extract potential filenames from task description
        file_patterns = re.findall(r"([a-zA-Z0-9_\-\./]+\.(?:md|txt|pdf|csv|py))", task_description)
        enhanced_desc = task_description
        
        for fp in file_patterns:
            if "@[" not in enhanced_desc:
                enhanced_desc = enhanced_desc.replace(fp, f"@[{fp}]")
        
        try:
            # Check if it's a simple text/md file and handle within RAG
            self.logger.info(f"[{self.name}] Analyzing context with Cowork logic.")
            # Use orchestrator's chat logic which already handles RAG (@[...])
            result = self.orchestrator.chat(enhanced_desc, verbose=False)
            self.logger.info(f"[{self.name}] Research completed.")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during research: {e}")
            raise e
