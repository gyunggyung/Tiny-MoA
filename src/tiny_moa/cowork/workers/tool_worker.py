from src.tiny_moa.cowork.workers.base import BaseWorker

class ToolWorker(BaseWorker):
    def __init__(self, name: str, logger, orchestrator):
        super().__init__(name, logger)
        self.orchestrator = orchestrator

    def execute(self, task_description: str, **kwargs) -> str:
        self.logger.info(f"[{self.name}] Tool processing: {task_description}")
        try:
            # Use the orchestrator's chat logic which already handles routing and tool calling
            # [Update] return_raw_tool_result=True ensures final integrator gets raw URLs/data
            result = self.orchestrator.chat(task_description, verbose=True, return_raw_tool_result=True)
            self.logger.info(f"[{self.name}] Tool task completed. Result: {str(result)[:50]}...")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] Error in ToolWorker: {e}")
            raise e
