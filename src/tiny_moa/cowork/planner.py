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
Your job is to break down a complex user goal into a sequence of simple, executable tasks.

AGENTS:
1. 'tool': Use for getting data, running commands, checking versions, or checking weather.
2. 'rag': Use ONLY for reading LOCAL files mentioned in context.
3. 'brain': Use for summarizing, explaining concepts, or reasoning.
4. 'office': Use for creating documents (PPT, Word, Excel). ALWAYS use this for presentations, reports, spreadsheets.

RULES for 'tool' tasks (Strict Format):
- "version" / "버전" -> "execute_command: <command> --version"
- "folder", "files", "list", "폴더", "파일" -> "execute_command: ls -R <folder>" (or dir). **USE EXACT FOLDER NAME.**
- "weather", "날씨" -> "get_weather: <city>"
- "news", "뉴스" -> "search_news: <keywords>" (Extract KEYWORDS ONLY. Remove "latest", "news abut", "find").
- "search", "검색" -> "search_web: <keywords>"
- **CRITICAL**: DO NOT use prefixes like "tool:", "command:", "도구:" inside description.
- **CRITICAL**: DO NOT TRANSLATE FOLDER NAMES or FILE NAMES.
- **CRITICAL**: If multiple requests, generated tasks for ALL parts.

RULES for 'office' tasks (Document Creation) - VERY IMPORTANT:
- "ppt", "powerpoint", "발표", "프레젠테이션", "슬라이드" -> Agent: office
- "word", "docx", "보고서", "문서", "제안서" -> Agent: office
- "excel", "xlsx", "엑셀", "스프레드시트", "통계", "표" -> Agent: office
- Description format: "create_ppt: <title> | <EXACT_FOLDER_NAME>" or "create_word: <title> | <EXACT_FOLDER_NAME>" or "create_excel: <topic> | <EXACT_FOLDER_NAME>"
- If user requests multiple document types, create a SEPARATE task for EACH type.
- **CRITICAL**: COPY THE EXACT FOLDER NAME FROM USER INPUT. DO NOT TRANSLATE OR MODIFY IT.
  - Example: User says "TestReports 폴더" -> Use "TestReports" (NOT "Reports")
  - Example: User says "Tiny-MoA-Korean-Reports" -> Use "Tiny-MoA-Korean-Reports" exactly

RULES for 'brain' tasks:
- "explain", "idea", "concept", "설명", "개념", "요약" -> Description: "Explain <topic>" (Agent: brain)

EXAMPLE INPUT: "트렌스포머 논문의 주요 아이디어 설명해줘. 그리고 uv 버전은? 딥마인드 뉴스도 찾아줘."
EXAMPLE OUTPUT:
[
  {{"description": "Explain the main idea of Transformer paper", "agent": "brain"}},
  {{"description": "execute_command: uv --version", "agent": "tool"}},
  {{"description": "search_news: DeepMind", "agent": "tool"}}
]

EXAMPLE INPUT: "tests 폴더에 뭐 있어? 런던과 서울 날씨 비교해줘. 앤트로픽 최신 소식은?"
EXAMPLE OUTPUT:
[
  {{"description": "execute_command: ls -R tests", "agent": "tool"}},
  {{"description": "get_weather: London", "agent": "tool"}},
  {{"description": "get_weather: Seoul", "agent": "tool"}},
  {{"description": "Compare weather of London and Seoul", "agent": "brain"}},
  {{"description": "search_news: Anthropic", "agent": "tool"}}
]

EXAMPLE INPUT: "Tiny-MoA 투자 제안서를 TestReports 폴더에 만들어줘. PPT, Word, Excel 모두 포함해."
EXAMPLE OUTPUT:
[
  {{"description": "create_ppt: Tiny-MoA 투자 제안서 | TestReports", "agent": "office"}},
  {{"description": "create_word: Tiny-MoA 투자 제안서 | TestReports", "agent": "office"}},
  {{"description": "create_excel: Tiny-MoA 통계 | TestReports", "agent": "office"}}
]

Context:
{context}

Goal: "{goal}"

Return ONLY the JSON list. No markdown."""
        
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
                plan = json.loads(json_str)

                # [Safety Fix] Force correct agent for known command prefixes
                # The 1.2B model sometimes assigns wrong agents.
                tool_prefixes = ["execute_command", "search_news", "search_web", "get_weather"]
                office_prefixes = ["create_ppt", "create_word", "create_excel"]
                
                for task in plan:
                    desc_lower = task.get("description", "").lower().strip()
                    
                    # Force tool agent
                    for prefix in tool_prefixes:
                        if desc_lower.startswith(prefix):
                            task["agent"] = "tool"
                            break
                    
                    # Force office agent
                    for prefix in office_prefixes:
                        if desc_lower.startswith(prefix):
                            task["agent"] = "office"
                            break
                
                return plan
            else:
                 # Fallback: create single task
                 return [{"description": user_goal, "agent": "brain"}]
        except json.JSONDecodeError:
            print(f"[Planner] Failed to parse plan JSON for goal: {user_goal}")
            return [{"description": user_goal, "agent": "brain"}] # Fallback
        except Exception as e:
            print(f"[Planner] Error in planning: {e}")
            return [{"description": user_goal, "agent": "brain"}] # Fallback
