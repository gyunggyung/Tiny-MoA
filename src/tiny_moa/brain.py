"""
Brain 모델 래퍼 (LiquidAI LFM2.5-1.2B)
=====================================
- 의도 분석
- 라우팅 결정
- 한국어 직접 처리
- 결과 통합
"""

import os
import re
from pathlib import Path
from typing import List, Optional
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
    "temperature": 0.05,  # [Critical] Thinking models require very low temp
    "top_k": 50,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
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
        self.n_ctx = n_ctx
        
        # logger.info(f"[Brain] Loaded! (threads={n_threads}, ctx={n_ctx})") # Removed print to clean UI
    
    def get_prompt_prefix(self) -> str:
        """Returns the prompt prefix (e.g. <|startoftext|>)"""
        return "<|startoftext|>"
    
    def route(self, user_input: str) -> dict:
        """
        사용자 입력을 분석하여 라우팅 결정
        
        Returns:
            {"route": "REASONER" | "DIRECT", "specialist_prompt": str}
        """
        user_lower = user_input.lower()
        
        # [Fast Path 0] 최신 정보 패턴 감지 (TOOL - search_web)
        # 연도(2023~2030), 버전(GPT-5, MoA 2.0, Claude 4), 최신 키워드
        # 지식의 한계를 미리 체크하여 LLM의 잘못된 판단 방지
        import re
        year_pattern = r'(202[3-9]|203[0-9])년?'
        version_pattern = r'(?:gpt|claude|moa|iphone|gemini|llama|mistral|qwen|v\.)[- ]?\d'
        recent_keywords = ["최신", "최근", "latest", "newest", "recent", "올해", "지난주", "어제"]
        
        if re.search(year_pattern, user_input) or re.search(version_pattern, user_lower) or any(k in user_lower for k in recent_keywords):
            return {"route": "TOOL", "specialist_prompt": user_input, "tool_hint": "search_web"}

        # [Fast Path 0.1] DIRECT 즉시 라우팅 (인사, 감사, 요약, 번역, 설명, 개념 질문)
        direct_fast = ["안녕", "hello", "hi ", "고마워", "감사", "thanks", "thank you", "반가워", "bye", "안녕히",
                      "요약해줘", "요약해", "정리해줘", "summarize", "summary", "번역해줘", "translate", 
                      "설명해줘", "explain", "차이점", "difference"]
        
        # "뭐야", "what is" 패턴: TOOL 키워드 없으면 DIRECT (개념 설명)
        concept_patterns = ["뭐야", "뭘까", "what is", "what's"]
        tool_keywords = ["날씨", "weather", "뉴스", "news", "검색", "search", "시간", "time", "버전", "version"]
        
        if any(k in user_lower for k in direct_fast):
            return {"route": "DIRECT", "specialist_prompt": "", "tool_hint": ""}
        
        # 개념 질문 (뭐야): 기술/도구 관련이면 TOOL(검색), 아니면 DIRECT
        if any(k in user_lower for k in concept_patterns):
            # 기술/도구 명칭이 있으면 검색이 필요 (TOOL)
            tech_terms = ["uv", "docker", "kubernetes", "npm", "pip", "git", "rust", "cargo", 
                         "langchain", "pytorch", "tensorflow", "react", "vue", "angular"]
            if any(t in user_lower for t in tech_terms) or not any(t in user_lower for t in tool_keywords):
                # 기술 용어가 있거나, 단순 개념 질문
                if any(t in user_lower for t in tech_terms):
                    return {"route": "TOOL", "specialist_prompt": user_input, "tool_hint": "search_web"}
                # 일반 개념 질문 (JSON이 뭐야?)
                if not any(t in user_lower for t in tool_keywords):
                    return {"route": "DIRECT", "specialist_prompt": "", "tool_hint": ""}
        
        # [Fast Path 0.5] TOOL 즉시 라우팅 (계산)
        calc_keywords = ["더해", "빼줘", "곱해", "나눠", "계산해", "calculate", "+", "-", "*", "/"]
        if any(k in user_lower for k in calc_keywords):
            return {"route": "TOOL", "specialist_prompt": user_input, "tool_hint": "calculate"}
        

        # [Fast Path 1] REASONER 즉시 라우팅 (코드, 알고리즘)
        reasoner_fast = ["함수 작성", "알고리즘 구현", "코드 작성", "피보나치", "fibonacci", "퀵소트", "quicksort", 
                        "aime", "문제 풀", "버그 찾", "디버깅", "debug", "최적화해줘", "optimize", "sql 쿼리"]
        if any(k in user_lower for k in reasoner_fast):
            return {"route": "REASONER", "specialist_prompt": user_input, "tool_hint": ""}
        
        # [Fast Path] 키워드 기반 즉시 라우팅 (LLM 호출 전)
        # 명백한 도구 요청("날씨", "버전 확인")은 LLM을 거치지 않고 바로 처리하여 속도/정확도 향상
        
        # 코딩/창작 관련 키워드가 있으면 Fast Path 건너뜀 (REASONER 가능성)
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
        
        # ChatML 포맷 수동 구성 (Official Template: <|startoftext|><|im_start|>system...)
        prefix = "<|startoftext|>"
        prompt = f"""{prefix}<|im_start|>system
{ROUTER_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        
        output = self.model(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>"],
            temperature=self.params["temperature"], # Use dynamic params
            top_p=self.params["top_p"],
            top_k=self.params["top_k"],
            repeat_penalty=self.params["repeat_penalty"],
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
    
    def route_pipeline(self, user_input: str) -> list:
        """
        다중 라우팅 파이프라인: 복합 작업을 여러 단계로 분해
        
        예: "최신 AI 트렌드 검색해서 요약해줘" 
            → [{"route": "TOOL", "tool_hint": "search_web", ...}, 
               {"route": "DIRECT", "task": "요약", ...}]
        
        Returns:
            list of routing decisions (순차 실행)
        """
        import re
        user_lower = user_input.lower()
        
        # ============================================
        # [Step 1] 복합 작업 패턴 감지
        # ============================================
        
        # 패턴: "~해서 ~해줘" (검색해서 요약해줘, 찾아서 설명해줘)
        # 주의: 단순 요청("알려줘")과 복합 요청("알려주고 판단해줘")을 구분해야 함
        compound_patterns = [
            # (TOOL 트리거, 후속 DIRECT 작업)
            (r'검색.{0,5}(요약|정리|설명|번역)', 'search_web', None),
            (r'찾아.{0,5}(요약|정리|설명|번역)', 'search_web', None),
            # 날씨: "알려주고 판단해" 같은 연결 패턴만 (단순 "알려줘"는 제외)
            (r'날씨.{0,10}(판단|추천|필요)', 'get_weather', None),
            (r'날씨.{0,5}알려.{0,5}(판단|추천|필요)', 'get_weather', None),
            (r'뉴스.{0,5}(요약|정리|브리핑)', 'search_news', None),
            (r'(버전|version).{0,10}(설명해)', 'search_web', None),
            # RAG + 날씨 복합 패턴: "문서 요약하고 날씨도 알려줘"
            (r'(요약|정리).{0,15}날씨.{0,5}(알려|확인)', 'get_weather', 'with_rag'),
            (r'날씨.{0,5}(알려|도).{0,10}(요약|정리)', 'get_weather', 'with_rag'),
        ]
        
        # 영어 패턴
        compound_patterns_en = [
            (r'search.{0,10}(summarize|explain|translate)', 'search_web', None),
            (r'find.{0,10}(summarize|explain|translate)', 'search_web', None),
            (r'weather.{0,10}(need|should|recommend)', 'get_weather', None),
            (r'news.{0,10}(summarize|brief)', 'search_news', None),
        ]
        
        all_patterns = compound_patterns + compound_patterns_en
        
        for pattern, tool_hint, _ in all_patterns:
            match = re.search(pattern, user_lower)
            if match:
                # 후속 작업 추출
                follow_up_task = match.group(1) if match.lastindex else "처리"
                
                # 파이프라인 생성
                pipeline = [
                    {
                        "route": "TOOL",
                        "specialist_prompt": user_input,
                        "tool_hint": tool_hint,
                        "step": 1,
                        "description": f"{tool_hint} 실행"
                    },
                    {
                        "route": "DIRECT",
                        "specialist_prompt": "",
                        "tool_hint": "",
                        "step": 2,
                        "description": f"결과 {follow_up_task}",
                        "context_from_step": 1  # Step 1의 결과를 컨텍스트로 사용
                    }
                ]
                return pipeline
        
        # ============================================
        # [Step 2] 복합 패턴 없으면 단일 라우팅
        # ============================================
        single_route = self.route(user_input)
        single_route["step"] = 1
        single_route["description"] = f"{single_route['route']} 단일 실행"
        return [single_route]

    def direct_respond(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """
        Brain이 직접 응답 (일반 대화, 한국어)
        """
        # 컨텍스트 초기화 (필수: 이전 상태가 남으면 decode 에러 발생)
        if hasattr(self.model, "reset"):
            self.model.reset()
        
        # ChatML 포맷 수동 구성
        # User requested specific default prompt: "You are a helpful assistant trained by Liquid AI."
        sys_content = system_prompt or "You are a helpful assistant trained by Liquid AI. Always respond in Korean unless asked otherwise."
        prefix = "<|startoftext|>"
        prompt = f"""{prefix}<|im_start|>system
{sys_content}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        
        # 직접 llm() 호출 (create_chat_completion 대신)
        output = self.model(
            prompt,
            max_tokens=self.n_ctx - 512, # Max context usage
            stop=["<|im_end|>"],
            temperature=self.params["temperature"],
            top_p=self.params["top_p"],
            top_k=self.params["top_k"],
            repeat_penalty=self.params["repeat_penalty"],
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
            # [Parsing Strategy]
            # input_data might be a single JSON string OR a multi-task Cowork format:
            # "[TASK: ...]\nDATA: {'...'} \n\n [TASK: ...]"
            
            import re
            
            sections = []
            # Check for Cowork format
            if "[TASK:" in specialist_output and "DATA:" in specialist_output:
                # Split by [TASK: ...] blocks
                raw_sections = re.split(r"\[TASK:.*?\]", specialist_output)
                for raw in raw_sections:
                    if "DATA:" in raw:
                        # Extract JSON part after "DATA:"
                        data_str = raw.split("DATA:", 1)[1].strip()
                        try:
                            data = eval(data_str)
                            sections.append(data)
                        except:
                            # If not a valid python dict/json, treat as plain text
                            # (e.g. Brain summary output)
                            if data_str:
                                sections.append({"type": "text", "content": data_str})
            else:
                # Try parsing as single JSON
                try:
                    data = eval(specialist_output) if "{" in specialist_output else {}
                    if isinstance(data, dict):
                         sections.append(data)
                except:
                    # Treat entire output as text if not JSON
                    sections.append({"type": "text", "content": specialist_output})

            # [Deterministic Formatting]
            final_formatted_blocks = []
            for data in sections:
                if not isinstance(data, dict): continue
                
                # Check for plain text wrapper
                if data.get("type") == "text" and "content" in data:
                    final_formatted_blocks.append(data["content"])
                    continue
                
                # Unwrap 'result' if present (Cowork Tool Result wrapper)
                # {'success': True, 'tool': 'search_news', 'result': {'results': [...]}}
                inner = data.get("result", data) 
                if not isinstance(inner, dict): inner = data # Fallback

                # 1. Search/News Results
                # Check both 'results' (direct) and 'inner["results"]'
                target_data = inner if "results" in inner else data
                
                if "results" in target_data and isinstance(target_data["results"], list):
                    block_lines = []
                    # Add query as header if available
                    q = target_data.get("query", "")
                    if q: block_lines.append(f"results for '{q}':")
                    
                    for item in target_data["results"]:
                        if isinstance(item, dict):
                            title = item.get("title", "No Title")
                            url = item.get("url", item.get("link", ""))
                            snippet = item.get("snippet", item.get("description", ""))
                            # Clean snippet
                            snippet = snippet.replace("\n", " ")[:200]
                            # Format: * Title
                            #           Summary...
                            #           Link: [Click to Read](URL)
                            # Using Markdown link syntax prevents long URL text from wrapping and breaking in TUI.
                            # Rich will render this as a clickable alias.
                            block_lines.append(f"* {title}\n  {snippet}\n  🔗 [Click to Read]({url})")
                    if block_lines:
                        final_formatted_blocks.append("\n".join(block_lines))
                        continue

                # 2. Weather Results
                # {'location': 'Seoul', 'temperature': ...}
                target_data = inner if "temperature" in inner else data
                if "temperature" in target_data and "condition" in target_data:
                    location = target_data.get("location", "City")
                    temp = target_data.get("temperature", "")
                    cond = target_data.get("condition", "")
                    final_formatted_blocks.append(f"* {location} Weather - {temp} / {cond}")
                    continue
                
                # 3. Fallback (Generic Dict)
                fallback_lines = []
                for k, v in target_data.items():
                    if isinstance(v, (str, int, float, bool)):
                        fallback_lines.append(f"- {k}: {v}")
                if fallback_lines:
                    final_formatted_blocks.append("\n".join(fallback_lines))

            if final_formatted_blocks:
                # If we achieved deterministic formatting, return it!
                # This bypasses the Hallucinating Brain.
                return "\n\n".join(final_formatted_blocks)

            # If formatting failed (empty), fallback to original string behavior (Legacy)
            # but usually sections would handle it.
            if not final_formatted_blocks and sections:
                 # Should not happen if sections populated, but just in case
                 formatted_output = str(sections)
        except Exception:
            pass # Continue to LLM if no deterministic output (unlikely for Search/Weather)

        # [English-First Strategy]
        # Generate in English first for speed and quality, then translate later.
        
        system_prompt = f"""You are a formatter. 
Your goal is to fill the provided data into the format below.

[STRICT FORMATTING RULES]
1. OUTPUT IN ENGLISH ONLY. Do NOT translate to Korean here.
2. Use the data provided in the 'Data' section.
3. OUTPUT MUST BE A BULLET LIST.
4. NO INTRO, NO OUTRO.
5. NEVER ALTER URLS. COPY THEM EXACTLY AS IS. Do not remove IDs or query parameters.

[TARGET FORMATS]
For WEATHER:
* City Weather - Temp / Condition
(Use data like 'temperature' and 'condition' from input)

For SEARCH/NEWS:
* Title - Summary (Link: URL)
!!! CRITICAL: YOU MUST INCLUDE THE FULL, EXACT URL FOR EVERY SEARCH RESULT !!!
Format: `* [Title] - [Summary] (Link: [URL])`
Example: `* AI News - content... (Link: https://example.com/article/ar-12345)`

[Data]
{formatted_output}

[User Request]
{user_input}

[Your Output]
""" 

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Output only the formatted list."},
            {"role": "user", "content": system_prompt},
        ]
        
        # [Stability Fix] Reset context
        if hasattr(self.model, "reset"):
            self.model.reset()
        
        # [Performance Optimization] Use INSTRUCT params (Fast, No Thinking)
        # We explicitly use LFM_INSTRUCT_PARAMS here regardless of self.use_thinking
        params = LFM_INSTRUCT_PARAMS.copy()
        
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=params.get("max_tokens", 4096), 
                **params,
            )
            
            return self._clean_response(response["choices"][0]["message"]["content"])
        except Exception as e:
            return f"Error integrating response: {e}"
    
    def _clean_response(self, text: str) -> str:
        """
        Thinking 모델의 <think>...</think> 태그를 제거하고 실제 응답만 추출합니다.
        태그가 닫히지 않은 경우(토큰 부족 등)에도 생각 부분을 최대한 제거합니다.
        """
        import re
        
        # 1. <think>... </think> 완벽한 태그 제거
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # 2. 닫는 태그가 잘린 경우 (<think>만 있고 </think>가 없음)
        if "<think>" in cleaned:
            # <think> 이후의 모든 내용을 생각 과정으로 간주하고 제거 (생각만 하다가 끝난 경우, 답변 없음)
            # 답변이 아예 없는 경우가 되므로, 에러 메시지 반환
            return "⚠️ 답변 생성 중 토큰 부족으로 중단되었습니다. (Thinking process truncated)"
            
        return cleaned

    def decompose_query(self, user_input: str) -> List[str]:
        """
        사용자의 복잡한 질문을 여러 개의 간단한 Tool 검색 쿼리로 분해합니다.
        v4: 휴리스틱 전용 (LLM 제거) - 속도 최적화 + 정확도 향상
        """
        import logging
        import re

        # [Step 0] 토픽 자동 감지 (정밀화 - 순서 중요!)
        topic = ""
        topic_keywords = {
            "날씨": ["날씨", "weather", "기온", "온도"],
            "뉴스": ["뉴스", "news", "기사", "article", "소식"],
            "주가": ["주가", "주식", "stock", "price"],
            "시간": ["시간", "time", "몇시"],
            "계산": ["더해", "빼", "곱해", "나눠", "계산", "calculate", "+", "-", "*", "/"],
        }
        
        # 토픽 감지 (가장 먼저 매칭되는 것 사용)
        for t, keywords in topic_keywords.items():
            if any(k in user_input.lower() for k in keywords):
                topic = t
                break

        # [Step 1] 비교/차이점 태스크 감지
        has_compare = any(k in user_input.lower() for k in ["비교", "compare", "vs", "차이", "difference"])
        
        # ===============================================
        # [v4] 휴리스틱 전용 (정밀 패턴 매칭)
        # ===============================================
        
        # Step 1: 다양한 연결어 패턴으로 분리
        # 한글: 과, 와, 랑, 이랑, 하고
        # 영어: and, or, vs, &
        # 기호: ,
        split_pattern = r"""
            (?<=[가-힣A-Za-z0-9])(?:과|와|랑|이랑|하고)\s*  |  # 한글 조사
            \s*,\s*  |                                        # 콤마
            \s+(?:그리고|and|or|vs|또는|&)\s+                  # 연결어
        """
        parts = re.split(split_pattern, user_input, flags=re.VERBOSE)
        
        # Step 2: 각 파트에서 핵심 엔티티 추출
        entities = []
        
        # 확장된 불용어
        stopwords = {
            # 한국어 동사/조사
            "날씨", "날씨를", "날씨와", "날씨는", "뉴스", "뉴스를", "검색", "검색해줘",
            "비교해봐", "비교", "알려줘", "해줘", "차이점", "차이", "보여줘",
            "그리고", "의", "을", "를", "가", "이", "는", "은", "에서", "으로", "에게",
            # 영어
            "weather", "news", "search", "compare", "difference", "tell", "show", "me", "the",
            "what", "is", "how", "about", "please", "in", "of", "to", "for", "a", "an",
        }
        
        # 토픽 키워드도 불용어에 추가
        for keywords in topic_keywords.values():
            for kw in keywords:
                stopwords.add(kw.lower())
        
        for part in parts:
            if not part:
                continue
            part = part.strip()
            
            # 공백으로 추가 분리
            words = part.split()
            for word in words:
                word_clean = word.strip()
                
                # 한국어 조사 제거 (긴 것부터)
                suffixes_ko = ["에서", "으로", "에게", "의", "를", "을", "이", "가", "은", "는"]
                for suffix in suffixes_ko:
                    if word_clean.endswith(suffix) and len(word_clean) > len(suffix) + 1:
                        word_clean = word_clean[:-len(suffix)]
                        break
                
                # 영어 소유격 제거
                if word_clean.endswith("'s"):
                    word_clean = word_clean[:-2]
                
                # 불용어 및 길이 체크
                if word_clean and word_clean.lower() not in stopwords and len(word_clean) >= 2:
                    # 숫자 처리: 계산 토픽일 때는 숫자 유지
                    if word_clean.isdigit() and topic != "계산":
                        continue
                    entities.append(word_clean)
        
        # 중복 제거 (순서 유지)
        entities = list(dict.fromkeys(entities))
        
        # [Step 3] 결과 생성
        if len(entities) >= 1:
            # 토픽 붙이기
            if topic:
                final_queries = [f"{ent} {topic}" for ent in entities]
            else:
                final_queries = entities.copy()
            
            # 비교 태스크 추가
            if has_compare and len(final_queries) >= 2:
                final_queries.append("Compare results")
                logging.info(f"[Brain] Added compare task")
            
            logging.info(f"[Brain] Heuristic v4: {final_queries}")
            return final_queries
        
        # Fallback: 원본 반환
        return [user_input]
        
        # ===============================================
        # [Fallback] LLM 분해 (휴리스틱 실패 시)
        # ===============================================
        try:
            # LFM2.5 Chat Template + Few-shot
            prompt = f"""<|startoftext|><|im_start|>system
You extract entities (cities, companies, topics) from queries. Return one entity per line. Do NOT include connectors or topic words.
<|im_end|>
<|im_start|>user
서울과 부산 날씨 비교해봐<|im_end|>
<|im_start|>assistant
서울
부산<|im_end|>
<|im_start|>user
삼성과 애플 뉴스 비교해봐<|im_end|>
<|im_start|>assistant
삼성
애플<|im_end|>
<|im_start|>user
React, Vue, Angular 차이점<|im_end|>
<|im_start|>assistant
React
Vue
Angular<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
            # 모델 리셋
            if hasattr(self.model, "reset"):
                self.model.reset()
            
            output = self.model(
                prompt,
                max_tokens=32,
                stop=["<|im_end|>", "\n\n"],
                temperature=0.1,  # LFM2.5 권장
                top_k=50,
                top_p=0.1,
                repeat_penalty=1.05,
                echo=False
            )
            content = output["choices"][0]["text"].strip()
            
            # 파싱
            llm_entities = []
            for line in content.split('\n'):
                clean = line.strip().lstrip('-*0123456789. ')
                if clean and len(clean) >= 2 and clean.lower() not in stopwords:
                    llm_entities.append(clean)
            
            if llm_entities:
                if topic:
                    final_queries = [f"{ent} {topic}" for ent in llm_entities]
                else:
                    final_queries = llm_entities
                
                if has_compare and len(final_queries) >= 2:
                    final_queries.append("Compare results")
                
                logging.info(f"[Brain] LLM v3 extracted: {final_queries}")
                return final_queries
                
        except Exception as e:
            logging.error(f"[Brain] LLM Decomposition failed: {e}")
        
        return [user_input]


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
