"""
Office Worker
=============
Brain이 내용을 생성하고 Office 도구를 호출하는 전문 Worker
"""

import json
import re
import os
from src.tiny_moa.cowork.workers.base import BaseWorker

# [Gemini-Claw Style] Senior Consultant Persona
OFFICE_SYSTEM_PROMPT = """You are an expert Business Consultant and Office Automation Specialist.
Your goal is to create highly professional, detailed, and insightful documents for the user.

[CORE PERSONA]
- Act as a Senior Consultant from a top-tier firm (e.g., McKinsey, BCG).
- Your output must be insightful, structured, and polished.
- Avoid generic or shallow content. Provide specific details based on the context.

[LANGUAGE RULES]
- IF THE USER ASKS IN KOREAN, YOU MUST GENERATE CONTENT IN KOREAN.
- Translate English context into Korean if necessary.
- Use professional business terminology.

[OUTPUT FORMAT]
- You must output ONLY valid JSON.
- Do not include markdown blocks (```json ... ```) or explanations.
- The JSON structure must match the user's request exactly.

[THINKING PROCESS]
1. Analyze the Request and Context.
2. Structure the document logically (Introduction -> Body -> Conclusion).
3. Draft content with high detail density.
4. Format as JSON.
"""

class OfficeWorker(BaseWorker):
    """Office 문서 생성 전문 Worker"""
    
    def __init__(self, name: str, logger, orchestrator):
        super().__init__(name, logger)
        self.orchestrator = orchestrator
    
    def execute(self, task_description: str, **kwargs) -> dict:
        """
        Office 문서 생성 실행
        """
        self.logger.info(f"[{self.name}] Office task: {task_description}")
        
        # 이전 단계의 실행 결과(Context) 가져오기
        context = kwargs.get("context", "")
        
        # [CRITICAL UPDATE] 로컬 프로젝트 정보 자동 주입
        # 사용자가 "우리 프로젝트"라고 했으므로, README.md를 읽어서 컨텍스트에 강제로 추가함
        try:
            readme_path = os.path.join(os.getcwd(), "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                    # 명확한 구분을 위해 헤더 추가
                    context += f"\n\n### 📂 PROJECT CONTEXT (Source of Truth: README.md)\n{readme_content[:4000]}\n[End of README]\n"
                    self.logger.info(f"[{self.name}] Auto-loaded README.md into context")
        except Exception as e:
            self.logger.warning(f"[{self.name}] Failed to read README.md: {e}")
        
        try:
            task_lower = task_description.lower()
            
            # 출력 폴더 추출 및 생성
            output_dir = self._get_output_dir(task_description)
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"[{self.name}] Output dir: {output_dir}")
            
            # 문서 유형 결정
            if any(k in task_lower for k in ["ppt", "powerpoint", "발표", "프레젠테이션", "슬라이드"]):
                return self._create_ppt(task_description, output_dir, context)
            elif any(k in task_lower for k in ["word", "docx", "보고서", "문서", "제안서"]):
                return self._create_word(task_description, output_dir, context)
            elif any(k in task_lower for k in ["excel", "xlsx", "엑셀", "스프레드시트", "표", "통계"]):
                return self._create_excel(task_description, output_dir, context)
            else:
                return self._create_all_documents(task_description, output_dir, context)
                
        except Exception as e:
            self.logger.error(f"[{self.name}] Office error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _get_output_dir(self, task_description: str) -> str:
        """출력 폴더 추출"""
        # | 구분자로 폴더명 추출 (Planner 형식: "create_ppt: 제목 | 폴더")
        if "|" in task_description:
            parts = task_description.split("|")
            if len(parts) >= 2:
                folder = parts[-1].strip()
                if folder:
                    return folder
        
        # 폴더명 추출 패턴들
        patterns = [
            r"['\"]([a-zA-Z가-힣0-9_-]+)['\"]?\s*(?:폴더|folder|directory)",
            r"(?:폴더|folder|directory)[:\s]+['\"]?([a-zA-Z가-힣0-9_-]+)['\"]?",
            r"(?:in|to|에)\s+['\"]?([a-zA-Z가-힣0-9_-]+)['\"]?\s*(?:폴더|folder)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, task_description, re.IGNORECASE)
            if match:
                return match.group(1)
        return "output"
    
    def _get_title(self, task_description: str) -> str:
        """제목 추출"""
        # | 구분자로 제목 추출
        if ":" in task_description:
            parts = task_description.split(":")
            if len(parts) >= 2:
                title_part = parts[1].split("|")[0].strip()
                if title_part:
                    return title_part
        return "Tiny-MoA 프로젝트"
    
    def _generate_content_with_brain(self, user_prompt: str, system_prompt: str = OFFICE_SYSTEM_PROMPT) -> str:
        """Brain을 사용하여 내용 생성 (System Prompt 분리)"""
        try:
            if hasattr(self.orchestrator, '_brain') and self.orchestrator._brain:
                # Brain.direct_respond에 system_prompt 전달
                response = self.orchestrator._brain.direct_respond(user_prompt, system_prompt=system_prompt)
                
                # [DEBUG] Brain 응답 로그
                self.logger.info(f"[{self.name}] Brain Response Length: {len(response) if response else 0}")
                if response:
                    self.logger.info(f"[{self.name}] Brain Response Preview: {response[:200]}...")
                
                if response and len(response) > 50:
                    return response
        except Exception as e:
            self.logger.warning(f"[{self.name}] Brain generation failed: {e}")
        return ""

    def _parse_json(self, content: str) -> dict:
        """JSON 파싱 시도 (Markdown 블록 지원)"""
        if not content:
            return {}
        try:
            # 1. Markdown code block 파싱
            code_block_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
            if code_block_match:
                return json.loads(code_block_match.group(1))
            
            # 2. 일반 JSON 파싱
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            self.logger.error(f"[{self.name}] JSON Decode Error: {e}")
            # [DEBUG] 실패한 내용 일부 출력
            self.logger.error(f"[{self.name}] Failed Content Preview: {content[:500]}")
        except Exception as e:
            self.logger.error(f"[{self.name}] Parse Error: {e}")
            
        return {}

    def _create_ppt(self, task_description: str, output_dir: str, context: str = "") -> dict:
        """PPT 생성"""
        title = self._get_title(task_description)
        
        # [Structured Prompt]
        user_prompt = f"""
[TASK]
Create a professional PowerPoint presentation about: {title}

[CONTEXT INFORMATION]
The following information is available. Use it to populate the slides with factual details.
{context}

[FORMAT REQUIREMENTS]
Generate a JSON object with 4-6 slides.
Structure:
{{
    "title": "Professional Title",
    "subtitle": "Insightful Subtitle",
    "slides": [
        {{"title": "Slide Title", "content": ["Detailed bullet point 1", "Detailed bullet point 2", "Key statistic or fact"]}},
        ...
    ]
}}

[IMPORTANT]
- Content MUST be in Korean (if query was Korean).
- Use bullet points for readability.
- Cover: Overview, Key Features, Architecture/Technology, Business Value, Future Roadmap.
"""

        content = self._generate_content_with_brain(user_prompt, system_prompt=OFFICE_SYSTEM_PROMPT)
        
        # JSON 파싱
        data = self._parse_json(content)
        if not data or "slides" not in data or not data["slides"]:
            self.logger.warning(f"[{self.name}] Brain failed to generate PPT content. Using fallback.")
            data = self._get_default_ppt_content(title)
        
        # OfficeAgent 사용
        from office.agent import OfficeAgent
        agent = OfficeAgent(workspace_root=".")
        
        output_path = os.path.join(output_dir, "presentation.pptx")
        result = agent.create_presentation(
            title=data.get("title", title),
            subtitle=data.get("subtitle", "Generated by Tiny-MoA"),
            slides=data.get("slides", []),
            output_path=output_path
        )
        
        self.logger.info(f"[{self.name}] PPT created: {result.get('path', 'unknown')}")
        return result
    
    def _create_word(self, task_description: str, output_dir: str, context: str = "") -> dict:
        """Word 문서 생성"""
        title = self._get_title(task_description)
        
        # [Structured Prompt]
        user_prompt = f"""
[TASK]
Create a comprehensive professional report (Word) about: {title}

[CONTEXT INFORMATION]
{context}

[FORMAT REQUIREMENTS]
Generate a JSON object with 4-6 detailed sections.
Structure:
{{
    "title": "Report Title",
    "sections": [
        {{"heading": "1. Executive Summary", "content": "High-level overview of the project/topic..."}},
        {{"heading": "2. Market/Problem Analysis", "content": "Detailed analysis..."}},
        {{"heading": "3. Solution Architecture", "content": "Technical details..."}},
        ...
    ]
}}

[CRITICAL]
- The main content MUST be in a list under the key "sections".
- Do NOT use keys like "key_points", "body", or "summary".
- JSON format only.

[IMPORTANT]
- Content MUST be in Korean.
- Use long paragraphs and bullet points (start with '- ').
- Be professional and thorough.
"""

        content = self._generate_content_with_brain(user_prompt, system_prompt=OFFICE_SYSTEM_PROMPT)
        data = self._parse_json(content)
        
        if not data or "sections" not in data or not data["sections"]:
            self.logger.warning(f"[{self.name}] Brain failed to generate Word content. Using fallback.")
            data = self._get_default_word_content(title)
        
        from office.agent import OfficeAgent
        agent = OfficeAgent(workspace_root=".")
        
        output_path = os.path.join(output_dir, "report.docx")
        result = agent.create_word_report(
            title=data.get("title", title),
            sections=data.get("sections", []),
            output_path=output_path
        )
        
        self.logger.info(f"[{self.name}] Word created: {result.get('path', 'unknown')}")
        return result
    
    def _create_excel(self, task_description: str, output_dir: str, context: str = "") -> dict:
        """Excel 생성"""
        title = self._get_title(task_description)
        
        # [Structured Prompt]
        user_prompt = f"""
[TASK]
Create an Excel spreadsheet with mock data analysis for: {title}

[CONTEXT INFORMATION]
{context}

[FORMAT REQUIREMENTS]
Generate a JSON object with 10-15 rows of realistic mock data.
Structure:
{{
    "sheet_name": "Analysis_Data",
    "data": [
        {{"Category": "Metric A", "Value": 100, "Growth": "5%"}},
        ...
    ]
}}
"""

        content = self._generate_content_with_brain(user_prompt, system_prompt=OFFICE_SYSTEM_PROMPT)
        data = self._parse_json(content)
        
        if not data or "data" not in data or not data["data"]:
            self.logger.warning(f"[{self.name}] Brain failed to generate Excel content. Using fallback.")
            data = self._get_default_excel_content(title)
        
        from office.agent import OfficeAgent
        agent = OfficeAgent(workspace_root=".")
        
        output_path = os.path.join(output_dir, "data.xlsx")
        result = agent.create_excel(
            data=data.get("data", []),
            output_path=output_path,
            sheet_name=data.get("sheet_name", "Data")
        )
        
        self.logger.info(f"[{self.name}] Excel created: {result.get('path', 'unknown')}")
        return result
    
    def _create_all_documents(self, task_description: str, output_dir: str, context: str = "") -> dict:
        """모든 문서 유형 생성"""
        results = {}
        results["ppt"] = self._create_ppt(task_description, output_dir, context)
        results["word"] = self._create_word(task_description, output_dir, context)
        results["excel"] = self._create_excel(task_description, output_dir, context)
        
        return {
            "success": True,
            "output_dir": output_dir,
            "documents": results,
            "message": f"Created PPT, Word, Excel in {output_dir}/"
        }
    

    
    def _get_default_ppt_content(self, title: str) -> dict:
        """기본 PPT 내용 (Brain 실패 시 사용)"""
        return {
            "title": title,
            "subtitle": "GPU Poor를 위한 AI 에이전트 프레임워크",
            "slides": [
                {
                    "title": "프로젝트 개요",
                    "content": [
                        "Tiny-MoA: 소형 모델의 집단 지성 활용",
                        "GPU 1개로도 GPT-4급 성능 달성 가능",
                        "로컬 LLM 기반 완전 프라이버시 보장",
                        "모듈형 아키텍처로 쉬운 확장성"
                    ]
                },
                {
                    "title": "핵심 기술",
                    "content": [
                        "Mixture of Agents (MoA) 아키텍처",
                        "LFM2.5-1.2B Brain 모델 (한국어 최적화)",
                        "RAG + CoWork 하이브리드 파이프라인",
                        "Intelligent Routing 시스템"
                    ]
                },
                {
                    "title": "시장 기회",
                    "content": [
                        "On-Device AI 시장 급성장",
                        "데이터 프라이버시 규제 강화",
                        "클라우드 AI 비용 부담 증가",
                        "엣지 컴퓨팅 수요 확대"
                    ]
                },
                {
                    "title": "경쟁 우위",
                    "content": [
                        "오픈소스: 커뮤니티 기여 가능",
                        "비용 효율성: 클라우드 대비 90% 절감",
                        "한국어 특화: 국내 시장 최적화",
                        "확장성: 다양한 도메인 적용 가능"
                    ]
                },
                {
                    "title": "투자 요청",
                    "content": [
                        "목표 투자금: 10억원",
                        "사용 계획: R&D 60%, 인력 30%, 마케팅 10%",
                        "예상 수익: 3년 내 BEP 달성",
                        "Exit 전략: 시리즈 A 또는 M&A"
                    ]
                }
            ]
        }
    
    def _get_default_word_content(self, title: str) -> dict:
        """기본 Word 내용 (Brain 실패 시 사용)"""
        return {
            "title": f"{title} 상세 제안서",
            "sections": [
                {
                    "heading": "1. 요약 (Executive Summary)",
                    "content": "Tiny-MoA는 GPU Poor 환경에서도 고성능 AI 에이전트를 구현할 수 있는 혁신적인 프레임워크입니다. 소형 언어 모델들의 집단 지성을 활용하여 GPT-4급 성능을 달성하면서도, 완전한 로컬 실행으로 데이터 프라이버시를 보장합니다. 본 프로젝트는 On-Device AI 시장의 성장과 함께 큰 잠재력을 가지고 있습니다."
                },
                {
                    "heading": "2. 문제 정의",
                    "content": "현재 AI 시장은 대형 클라우드 모델에 과도하게 의존하고 있습니다. 이로 인해 높은 API 비용, 데이터 유출 위험, 네트워크 지연 문제가 발생합니다. 특히 기업 환경에서 민감한 데이터를 클라우드로 전송하는 것은 심각한 보안 리스크를 초래합니다. Tiny-MoA는 이러한 문제를 근본적으로 해결합니다."
                },
                {
                    "heading": "3. 솔루션",
                    "content": "Tiny-MoA는 Mixture of Agents 아키텍처를 기반으로 합니다. Brain(지휘), Tool(실행), RAG(검색), Writer(작성) 등 전문화된 에이전트들이 협력하여 복잡한 작업을 수행합니다. 1.2B 파라미터의 소형 모델로도 뛰어난 성능을 달성하며, 완전한 로컬 실행으로 프라이버시를 보장합니다."
                },
                {
                    "heading": "4. 비즈니스 모델",
                    "content": "오픈소스 기반으로 커뮤니티 생태계를 구축하고, 엔터프라이즈 라이선스와 기술 지원으로 수익을 창출합니다. 추가로 도메인 특화 모델 Fine-tuning 서비스와 컨설팅을 제공합니다. 클라우드 AI 대비 90% 이상의 비용 절감 효과로 고객 확보가 용이합니다."
                },
                {
                    "heading": "5. 팀 구성",
                    "content": "핵심 팀원은 AI/ML 분야 5년 이상 경력의 전문가들로 구성되어 있습니다. 오픈소스 프로젝트 기여 경험이 풍부하며, 국내외 AI 컨퍼런스 발표 실적이 있습니다. 스타트업 운영 경험을 바탕으로 빠른 실행력을 갖추고 있습니다."
                },
                {
                    "heading": "6. 재무 계획",
                    "content": "1년차에는 R&D에 집중하여 핵심 기술을 완성합니다. 2년차에는 베타 고객 확보와 함께 매출을 시작합니다. 3년차에는 BEP 달성을 목표로 하며, 이후 시리즈 A 투자 유치 또는 전략적 M&A를 통한 Exit을 계획하고 있습니다."
                }
            ]
        }
    
    def _get_default_excel_content(self, title: str) -> dict:
        """기본 Excel 내용 (Brain 실패 시 사용)"""
        return {
            "sheet_name": "모듈 통계",
            "data": [
                {"모듈명": "Brain", "파일수": 5, "코드라인": 1200, "상태": "완료", "담당자": "김개발"},
                {"모듈명": "Tool Executor", "파일수": 3, "코드라인": 800, "상태": "완료", "담당자": "이엔지"},
                {"모듈명": "RAG System", "파일수": 4, "코드라인": 600, "상태": "완료", "담당자": "박데이터"},
                {"모듈명": "CoWork Framework", "파일수": 8, "코드라인": 1500, "상태": "완료", "담당자": "최아키"},
                {"모듈명": "Office Agent", "파일수": 3, "코드라인": 400, "상태": "완료", "담당자": "정오피스"},
                {"모듈명": "TUI Dashboard", "파일수": 2, "코드라인": 500, "상태": "완료", "담당자": "한UI"},
                {"모듈명": "Orchestrator", "파일수": 1, "코드라인": 1800, "상태": "완료", "담당자": "최아키"},
                {"모듈명": "Planner", "파일수": 1, "코드라인": 200, "상태": "완료", "담당자": "김개발"},
                {"모듈명": "Memory System", "파일수": 2, "코드라인": 300, "상태": "개발중", "담당자": "이엔지"},
                {"모듈명": "Doc Processor", "파일수": 3, "코드라인": 450, "상태": "완료", "담당자": "박데이터"},
                {"합계": "-", "파일수": 32, "코드라인": 7750, "상태": "-", "담당자": "-"}
            ]
        }
