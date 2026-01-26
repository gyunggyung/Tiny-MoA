"""
Planner Agent
=============
사용자의 모호한 요청을 구체적인 실행 계획(Task List)으로 변환.
Brain 모델을 사용하여 JSON 형식의 계획을 생성합니다.
"""

from typing import List
from src.tiny_moa.brain import Brain
from src.tiny_moa.cowork.task_queue import CoworkTask, TaskQueue

class PlannerAgent:
    def __init__(self, brain: Brain):
        self.brain = brain
        
    def create_plan(self, user_goal: str, context_str: str) -> List[dict]:
        """
        사용자 목표를 분석하여 태스크 리스트 생성
        Returns:
            List[dict]: [{"description": "...", "agent": "..."}]
        """
        
        system_prompt = """You are a Task Planner for an autonomous AI coworker.
Your job is to break down a high-level goal into a sequence of concrete, executable tasks.
The available agents are:
- 'rag': ONLY for searching or reading local FILES (PDF, Markdown, Docs). Do NOT use if the goal doesn't mention files.
- 'brain': Summarizing, Writing, General reasoning, Addressing the user directly.
- 'tool': External data (Weather, Web search), Shell commands.

Context:
{context}

Goal: "{goal}"


IMPORTANT:
- If the goal is a simple question (e.g., weather, time, greetings), use only 1 or 2 tasks.
- DO NOT use 'rag' unless specifically asked to read a file or search the workspace for data.

Return a LIST of tasks in JSON format. Example:
[
  {{"description": "List all files in downloads folder", "agent": "tool"}},
  {{"description": "Extract text from report.pdf", "agent": "rag"}},
  {{"description": "Summarize the extracted text", "agent": "brain"}}
]
Return ONLY the JSON list. No markdown, no explanation."""
        
        prompt = system_prompt.format(context=context_str, goal=user_goal)
        
        response = self.brain.direct_respond(prompt, system_prompt="You are a JSON generator.")
        
        # Clean up response
        import json
        import re
        
        try:
            # Markdown block removal
            cleaned = response.replace("```json", "").replace("```", "").strip()
            # Find list bracket
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start != -1 and end != -1:
                json_str = cleaned[start:end]
                tasks = json.loads(json_str)
                return tasks
            else:
                 # Fallback: create single task
                 return [{"description": user_goal, "agent": "brain"}]
        except Exception as e:
            print(f"[Planner] Error parsing plan: {e}")
            # Fallback
            return [{"description": user_goal, "agent": "brain"}]
