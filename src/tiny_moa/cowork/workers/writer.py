from src.tiny_moa.cowork.workers.base import BaseWorker
import re

class WriterWorker(BaseWorker):
    def __init__(self, name: str, logger, brain, file_skill):
        super().__init__(name, logger)
        self.brain = brain
        self.file_skill = file_skill

    def execute(self, task_description: str, **kwargs) -> str:
        self.logger.info(f"[{self.name}] Starting writing task: {task_description}")
        
        history = kwargs.get("history", "")
        user_goal = kwargs.get("user_goal", "")
        
        summary_prompt = f"""You are a Professional Writer.
Goal: {user_goal}

Previous Context/Results:
{history}

Current Task: {task_description}

Write a high-quality, comprehensive final report or content based on the above.
IMPORTANT: The final report MUST be written in KOREAN (한국어).
Return ONLY the content to be saved."""

        try:
            result = self.brain.direct_respond(summary_prompt)
            
            # Find target filename in task description
            file_patterns = re.findall(r"([a-zA-Z0-9_\-\./]+\.(?:md|txt|pdf|csv))", task_description)
            target_file = "docs/cowork_result.md" # Default
            if file_patterns:
                target_file = file_patterns[0]
            
            self.logger.info(f"[{self.name}] Saving result to {target_file}")
            self.file_skill.execute_tool("workspace_write", {"filename": target_file, "content": result})
            
            self.logger.info(f"[{self.name}] Writing task completed.")
            return f"Saved to {target_file}"
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during writing: {e}")
            raise e
