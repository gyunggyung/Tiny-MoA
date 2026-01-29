from src.tiny_moa.cowork.workers.base import BaseWorker

class BrainWorker(BaseWorker):
    def __init__(self, name: str, logger, brain):
        super().__init__(name, logger)
        self.brain = brain

    def execute(self, task_description: str, history: str = "", **kwargs) -> str:
        self.logger.info(f"[{self.name}] Brain processing: {task_description}")
        
        # Prepend history if available to provide context for the Brain
        full_prompt = task_description
        if history:
            full_prompt = f"Previous Task Results:\n{history}\n\nCurrent Task: {task_description}\n\nIMPORTANT: Please perform the current task using the provided context above. ALWAYS respond in KOREAN."

        try:
            result = self.brain.direct_respond(full_prompt)
            self.logger.info(f"[{self.name}] Brain task completed. Result len: {len(str(result))}")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] Error in BrainWorker: {e}")
            raise e
