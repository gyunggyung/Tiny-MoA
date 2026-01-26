"""
Brain 모델 래퍼 (LiquidAI LFM2.5-1.2B)
=====================================
- 의도 분석
- 라우팅 결정
- 한국어 직접 처리
- 결과 통합
"""

import os
from pathlib import Path
from typing import Optional
from llama_cpp import Llama
import sys
import logging

# [Optimization] Silence llama-cpp logs to keep UI clean
os.environ["LLAMA_CPP_LOG_LEVEL"] = "error" 
logging.getLogger("llama_cpp").setLevel(logging.ERROR)

# LFM2.5 권장 파라미터 (공식 문서: docs.liquid.ai/lfm/inference/llama-cpp)
LFM_INSTRUCT_PARAMS = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

LFM_THINKING_PARAMS = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# 라우터 시스템 프롬프트
ROUTER_SYSTEM_PROMPT = """You are a task router. Analyze the user's request and decide how to handle it.

Available specialists:
- REASONER: STRICTLY for pure coding tasks (writing Python functions/classes) and complex algorithmic/math problems only. Do NOT use for "summarizing", "explaining", "reading files", "checking versions", or "general info".
- TOOL: For requests requiring external information (weather, news, definitions), system status, verify commands, or real-time data.
- DIRECT: For general conversation, summaries, explanations, greetings, translations, and internal knowledge.

Respond with a JSON object:
{"route": "REASONER" or "TOOL" or "DIRECT", "specialist_prompt": "optimized search keywords for TOOL. For 'execute_command', provide the EXACT shell command. Do NOT provide natural language descriptions.", "tool_hint": "tool name if TOOL route"}

Examples:
- "피보나치 함수 작성해줘" → {"route": "REASONER", "specialist_prompt": "Write a Python function for Fibonacci sequence", "tool_hint": ""}
- "이 문서 요약해줘" → {"route": "DIRECT", "specialist_prompt": "", "tool_hint": ""}
- "서울 날씨 어때?" → {"route": "TOOL", "specialist_prompt": "Seoul", "tool_hint": "get_weather"}
- "uv가 뭐야?" → {"route": "TOOL", "specialist_prompt": "what is uv python tool", "tool_hint": "search_web"}
"""


class Brain:
    """LFM2.5-1.2B 기반 Brain 모델"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        use_thinking: bool = False,  # PoC에서 실험 후 결정
    ):
        """
        Args:
            model_path: GGUF 모델 경로. None이면 기본 경로 사용
            n_ctx: 컨텍스트 길이
            n_threads: CPU 스레드 수. None이면 자동 감지
            use_thinking: Thinking 모델 사용 여부 (실험 중)
        """
        self.use_thinking = use_thinking
        self.params = LFM_THINKING_PARAMS if use_thinking else LFM_INSTRUCT_PARAMS
        
        # 모델 경로 결정
        if model_path is None:
            # 1. 로컬 models/ 폴더 확인
            base_dir = Path(__file__).parent.parent.parent / "models" / "brain"
            gguf_files = list(base_dir.glob("*.gguf")) if base_dir.exists() else []
            
            if gguf_files:
                model_path = str(gguf_files[0])
            else:
                # 2. HuggingFace 캐시에서 자동 다운로드/찾기
                try:
                    from huggingface_hub import hf_hub_download
                    model_name = "LFM2.5-1.2B-Thinking-Q4_K_M.gguf" if use_thinking else "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
                    repo_id = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF" if use_thinking else "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
                    model_path = hf_hub_download(repo_id=repo_id, filename=model_name)
                except Exception as e:
                    raise FileNotFoundError(
                        f"모델을 찾을 수 없습니다. 다운로드해주세요:\n"
                        f"huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf\n"
                        f"Error: {e}"
                    )
        
        # logger.info(f"[Brain] Loading model from: {model_path}") # Removed print to clean UI
        
        # 스레드 수 결정 (CPU 코어의 절반 권장)
        if n_threads is None:
            n_threads = max(1, os.cpu_count() // 2)
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        
        # logger.info(f"[Brain] Loaded! (threads={n_threads}, ctx={n_ctx})") # Removed print to clean UI
    
    def route(self, user_input: str) -> dict:
        """
        사용자 입력을 분석하여 라우팅 결정
        
        Returns:
            {"route": "REASONER" | "DIRECT", "specialist_prompt": str}
        """
        user_lower = user_input.lower()
        
        # [Fast Path] 키워드 기반 즉시 라우팅 (LLM 호출 전)
        # 명백한 도구 요청("날씨", "버전 확인")은 LLM을 거치지 않고 바로 처리하여 속도/정확도 향상
        
        # 1. 코딩/창작 관련 키워드가 있으면 Fast Path 건너뜀 (REASONER 가능성)
        creation_keywords = ["write", "code", "create", "generate", "function", "script", "class", "impl", "작성", "만들", "구현", "짜줘"]
        is_creation = any(k in user_lower for k in creation_keywords)
        
        if not is_creation:
            # TOOL 키워드 매칭
            fast_tools = {
                "get_weather": ["날씨", "weather", "기온", "온도"],
                "search_web": ["검색", "search", "정보", "info", "search_web"],
                "search_news": ["뉴스", "news", "최신", "기사", "article", "소식", "보도", "발표", "기사들", "search_news"],
                "execute_command": ["version", "버전", "check", "확인", "실행", "run", "installed", "설치", "status", "환경"],
                "get_current_time": ["시간", "time", "몇시", "date", "오늘"],
            }
            
            # [Historical Data Fallback]
            # wttr.in은 과거 데이터를 지원하지 않으므로, 과거 관련 키워드가 있으면 검색으로 유도
            historical_keywords = ["yesterday", "last week", "history", "past", "어제", "지난", "과거", "작년"]
            is_historical = any(k in user_lower for k in historical_keywords)

            for tool_name, keywords in fast_tools.items():
                if any(kw in user_lower for kw in keywords):
                    # 날씨 조회인데 과거 데이터라면 -> Search Web으로 변경
                    if tool_name == "get_weather" and is_historical:
                        return {"route": "TOOL", "specialist_prompt": user_input, "tool_hint": "search_web"}

                    # execute_command의 경우 추가 검증
                    if tool_name == "execute_command":
                        # "python version", "check uv" 등은 확실한 명령
                        cmd_targets = ["python", "uv", "pip", "node", "npm", "git", "docker", "system", "os"]
                        if any(t in user_lower for t in cmd_targets) or "ls" in user_lower or "dir" in user_lower:
                             # Argument는 Orchestrator/Falcon에게 위임 ("" 전달)
                              return {"route": "TOOL", "specialist_prompt": user_input, "tool_hint": tool_name}
                    else:
                        # Argument는 Orchestrator/Falcon에게 위임 ("" 전달)
                        # 예: "서울 날씨" -> Prompt="" -> Falcon이 "Seoul" 추출
                        return {"route": "TOOL", "specialist_prompt": user_input, "tool_hint": tool_name}

        # 컨텍스트 초기화
        if hasattr(self.model, "reset"):
            self.model.reset()
        
        # ChatML 포맷 수동 구성
        prompt = f"""<|im_start|>system
{ROUTER_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        
        output = self.model(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>"],
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        content = output["choices"][0]["text"].strip()
        
        # JSON 파싱 시도
        try:
            import json
            # JSON 부분만 추출
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # [Fast Path] DIRECT 키워드 체크 (강력 추천)
        direct_keywords = ["요약", "정리", "설명", "summarize", "explain", "translate", "번역", "안녕", "hello", "hi", "반가워"]
        if any(kw in user_lower for kw in direct_keywords) and not is_creation:
             return {"route": "DIRECT", "specialist_prompt": "", "tool_hint": ""}

        # REASONER 키워드 (순수 코딩만)
        keywords_reasoner = ["함수", "알고리즘", "수학", "증명", "aime", "fibonacci", "script", "class"]
        
        # 'python'이나 '코드'가 있으면 REASONER 가능성 높음
        if ("python" in user_lower or "코드" in user_lower or "code" in user_lower) and not any(k in user_lower for k in ["version", "check", "확인", "버전", "summarize", "요약"]):
             return {"route": "REASONER", "specialist_prompt": user_input, "tool_hint": ""}
             
        if any(kw in user_lower for kw in keywords_reasoner) and not any(kw in user_lower for kw in direct_keywords):
            return {"route": "REASONER", "specialist_prompt": user_input, "tool_hint": ""}
        
        return {"route": "DIRECT", "specialist_prompt": "", "tool_hint": ""}
    
    def direct_respond(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """
        Brain이 직접 응답 (일반 대화, 한국어)
        """
        # 컨텍스트 초기화 (필수: 이전 상태가 남으면 decode 에러 발생)
        if hasattr(self.model, "reset"):
            self.model.reset()
        
        # ChatML 포맷 수동 구성
        sys_content = system_prompt or "You are a helpful assistant. Always respond in Korean unless asked otherwise."
        prompt = f"""<|im_start|>system
{sys_content}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        
        # 직접 llm() 호출 (create_chat_completion 대신)
        output = self.model(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>"],
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        return output["choices"][0]["text"].strip()
    
    def integrate_response(self, user_input: str, specialist_output: str) -> str:
        """
        Specialist 출력을 사용자에게 맞게 통합/포맷팅
        """
        # Tool output이 dict string일 경우 보기 좋게 변환 시도
        formatted_output = specialist_output
        try:
            import json
            if isinstance(specialist_output, str) and "{" in specialist_output:
                # 작은 모델은 JSON보다 Key-Value 리스트를 더 잘 이해함
                data = eval(specialist_output) if "{" in specialist_output else {} # safe eval for dict string
                if isinstance(data, dict):
                    lines = []
                    for k, v in data.items():
                        lines.append(f"- {k}: {v}")
                    formatted_output = "\n".join(lines)
        except:
            pass

        system_prompt = """You are a Professional Data Integration Assistant.
Your goal is to synthesize multiple task results into a single, comprehensive Korean report.

YOUR JOB:
1. READ ALL the facts below. 
2. DO NOT skip any data point. If multiple items (e.g. news articles) are listed, YOU MUST report on all of them.
3. [CRITICAL] YOU MUST INCLUDE THE ACTUAL URLs/LINKS in the report. DO NOT USE PLACEHOLDERS like [link]. COPY THE URL from the facts below.
4. If a tool failed or data is missing for some items, explicitly state that.
5. ALWAYS write the final response in KOREAN (한국어).

FACTS:
""" + formatted_output

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_input}\n\nPlease write the answer response:"},
        ]
        
        # [Stability Fix] 컨텍스트 초기화
        # 통합 단계는 독립적이므로 이전 대화 맥락이 필요 없음 (Facts에 다 있음)
        if hasattr(self.model, "reset"):
            self.model.reset()
        
        # Temperature 0.1로 창의성 억제
        params = self.params.copy()
        params["temperature"] = 0.1
        
        try:
            response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=512,
            **params,
        )
        
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error integrating response: {e}"
    
    def decompose_query(self, user_input: str) -> list[str]:
        """
        복합 질문을 단순 질문 리스트로 분해
        예: "서울과 도쿄 날씨 비교해줘" -> ["서울 날씨 어때?", "도쿄 날씨 어때?", "두 날씨 비교해줘"]
        """
        system_prompt = """You are a query decomposer.
Your task is to split a complex question into simple sub-questions.

OUTPUT FORMAT:
Provide a pure list of queries, one per line. Start each line with a hyphen "- ".
NO markdown, NO explanations.

EXAMPLES:
Input: "Compare Seoul and Tokyo weather"
Output:
- Get weather in Seoul
- Get weather in Tokyo

Input: "Check python and uv versions"
Output:
- check python version
- check uv version
"""
        # [Stability Fix] LFM 1.2B 모델이 JSON/List 생성 시 'llama_decode returned -1' 크래시가 잦음
        # 따라서 LLM 호출을 건너뛰고 바로 휴리스틱 분해를 시도함
        # 필요한 경우에만 LLM을 켜도록 플래그 처리 가능하나, 현재 환경에서는 안정성 최우선.
        pass
        
        """
        content = ""
        try:
            # (LLM Call skipped to prevent crash)
            pass
        except:
             pass
        """
            
        # [Fallback] 휴리스틱/Regex 분해 (LLM 실패 시)
        # "서울과 도쿄" -> ["서울", "도쿄"] -> ["서울 날씨", "도쿄 날씨"] (날씨가 포함된 경우)
        import re
        topic = ""
        if any(k in user_input for k in ["날씨", "weather"]):
            topic = "날씨"
        elif any(k in user_input for k in ["뉴스", "news", "기사", "article", "소식"]):
            topic = "뉴스"
        
        # Regex로 분리 (와/과/랑/이랑/vs/and/,)
        # \s*는 공백이 있을수도 없을수도 있음을 의미
        # (?: ... )는 비캡처 그룹
        # Regex로 분리 (와/과/랑/이랑/vs/and/,)
        # \s*는 공백이 있을수도 없을수도 있음을 의미
        # (?: ... )는 비캡처 그룹
        # ? 문자로도 분리 (질문이 여러 개인 경우)
        # Regex로 분리 (와/과/랑/이랑/vs/and/,)
        # [Fix] More inclusive pattern for connectors
        split_pattern = r"\s+(?:vs|and|&|or|또는|그리고|와|과|랑|이랑)\s+|\s*,\s*|\s*\?\s*"
        
        parts = re.split(split_pattern, user_input)
        
        # 정제 및 유효성 검사
        parts = [p.strip() for p in parts if len(p.strip()) > 1]
        
        if len(parts) > 1:
            # 정제된 쿼리 생성
            final_queries = []
            for p in parts:
                # 불필요한 서술어 제거 (비교해줘, 알려줘 등)
                # 주의: "어때" 뒤에 오는 내용이 삭제되면 안되므로 .* 사용 시 주의.
                # 이미 분리되었으므로 p는 "서울 날씨 어때" 형태일 것임. 따라서 .* 써도 됨.
                clean_p = re.sub(r"(비교|compare|알려줘|해줘|어때|Check|Verify|with).*", "", p).strip()
                
                if not clean_p: continue
                
                # "도쿄 날씨" 처럼 날씨/뉴스가 이미 포함된 경우 중복 방지
                if topic and topic not in clean_p and "날씨" not in clean_p and "뉴스" not in clean_p and "기사" not in clean_p:
                     q = f"{clean_p} {topic}".strip()
                else:
                     q = clean_p
                
                if len(q) > 1: # 너무 짧은 쿼리 제외
                     final_queries.append(q)
            
            if len(final_queries) > 1:
                print(f"[Brain] Heuristic Decomposition: {final_queries}")
                return final_queries
            
        return [user_input] # 실패 시 원본 그대로 반환


if __name__ == "__main__":
    # 테스트
    print("=== Brain 테스트 ===")
    brain = Brain()
    
    # 라우팅 테스트
    test_inputs = [
        "피보나치 함수 작성해줘",
        "안녕하세요!",
        "1 + 1 = ?",
        "AIME 2024 문제를 풀어봐",
    ]
    
    for inp in test_inputs:
        result = brain.route(inp)
        print(f"Input: {inp}")
        print(f"Route: {result}")
        print()
