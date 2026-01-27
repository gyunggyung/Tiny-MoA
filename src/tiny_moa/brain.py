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
                        json_str = raw.split("DATA:", 1)[1].strip()
                        try:
                            data = eval(json_str)
                            sections.append(data)
                        except:
                            pass
            else:
                # Try parsing as single JSON
                try:
                    data = eval(specialist_output) if "{" in specialist_output else {}
                    if isinstance(data, dict):
                         sections.append(data)
                except:
                    pass

            # [Deterministic Formatting]
            final_formatted_blocks = []
            for data in sections:
                if not isinstance(data, dict): continue
                
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
        LLM을 우선 사용하고, 실패 시 정규식/휴리스틱으로 fallback합니다.
        """
        import logging
        from typing import List, Optional
        import re # Ensure re is imported if not already

        try:
            # LLM Prompt for Decomposition
            # [Critical Fix] Prevent splitting simple tasks into steps. 
            # We want Tool PARALLELIZATION, not Step-by-Step planning.
            prompt = f"""<|startoftext|>
Task: detailed analysis of whether to split the query.
Query: "{user_input}"

Rules:
1. ONLY split if the user asks for TWO DIFFERENT things (e.g. "Seoul AND Tokyo").
2. Do NOT split a single request into "Check" and "Provide". That is redundant.
3. If it's a single location/topic, return the original query as the ONLY line.

Examples:
"Seoul weather" ->
Seoul weather

"Seoul and Tokyo weather" ->
Seoul weather
Tokyo weather

"Check weather in Seoul" ->
Check weather in Seoul

"Seoul weather?" ->
Seoul weather

User: "{user_input}"
Result:
"""
            # Timeout/Crash 방지를 위한 파라미터 튜닝
            output = self.model(
                prompt,
                max_tokens=128, 
                stop=["<|im_end|>", "\n\n", "User:", "Task:", "Result:"], 
                temperature=0.0, 
                echo=False
            )
            content = output["choices"][0]["text"].strip()
            
            # Robust Parsing
            lines = []
            for line in content.split('\n'):
                # 숫자, 불렛, 하이픈 등 제거
                clean_line = re.sub(r"^[\d\-\*\.]+\s*", "", line.strip()).strip()
                if len(clean_line) > 1: 
                     lines.append(clean_line)
            
            # [Aggressive Filter] If decomposition results in more lines, check if they are just synonyms
            if len(lines) > 1:
                # If original query was short (< 5 words), decomposition is risky unless it has "and/,"
                if len(user_input.split()) < 5 and not any(k in user_input for k in ["and", ",", "와", "과", "하고", "vs"]):
                     logging.warning(f"[Brain] Decomposition rejected (Short query, no explicit separator): {lines}")
                     return [user_input]
                     
                logging.info(f"[Brain] LLM Decomposition Success: {lines}")
                return lines
            else:
                 # [Improvement] If LLM failed to split (returned 1 line), 
                 # BUT we detected complex keywords/separators, fall through to Regex/Heuristic below.
                 # Do NOT return [user_input] immediately.
                 logging.info("[Brain] LLM returned 1 line, trying fallback heuristic...")
                 pass 

                
        except Exception as e:
            logging.error(f"[Brain] LLM Decomposition failed: {e}")
            pass
            
        # [Fallback] 휴리스틱/Regex 분해 (LLM 실패 시)
        # "서울과 도쿄" -> ["서울", "도쿄"] -> ["서울 날씨", "도쿄 날씨"] (날씨가 포함된 경우)
        import re
        topic = ""
        if any(k in user_input for k in ["날씨", "weather", "기온", "온도"]):
            topic = "날씨"
        elif any(k in user_input for k in ["뉴스", "news", "기사", "article", "소식"]):
            topic = "뉴스"
        
        # Regex로 분리 (와/과/랑/이랑/vs/and/,)
        # [Fix] More inclusive pattern for connectors
        split_pattern = r"(?<=[가-힣])(?:과|와|랑|이랑)\s+|\s+(?:vs|and|&|or|또는|그리고)\s+|\s*,\s*|\s*\?\s*"
        
        parts = re.split(split_pattern, user_input)
        
        # [Fallback Enhancement] "광주 춘천 날씨" -> ["광주", "춘천", "날씨"] -> ["광주 날씨", "춘천 날씨"]
        # LLM 실패 시, 단순 공백으로도 분리 시도
        final_parts = []
        for part in parts:
            part = part.strip()
            if not part: continue
            
            # 1. 이미 완성된 문장이면 패스
            if len(part.split()) > 3: 
                final_parts.append(part)
                continue
                
            # 2. 공백으로 나눴을 때 2개 이상이고, "날씨" 같은 키워드가 포함된 경우
            sub_parts = part.split()
            if len(sub_parts) >= 2 and any(k in part for k in ["날씨", "weather", "기온"]):
                # 마지막 단어가 공통 키워드일 확률 높음 (예: "서울 대전 날씨")
                keyword = sub_parts[-1]
                # "날씨를", "날씨는" 등 조사가 붙어도 처리가능하도록 수정
                if any(x in keyword for x in ["날씨", "weather", "기온", "온도"]):
                    for sub in sub_parts[:-1]:
                        final_parts.append(f"{sub} {keyword}")
                else:
                    # 키워드가 명확지 않으면 그냥 다 넣음
                    final_parts.extend(sub_parts)
            else:
                 final_parts.append(part)

        parts = final_parts
        
        # [Filtering] 불용어 및 무의미한 조각 제거
        filtered_parts = []
        for p in parts:
            p_clean = p.strip()
            # 1. 너무 짧거나 특수문자만 있는 경우 제외
            if len(p_clean) < 2: 
                continue
            # 2. "날씨를", "비교해줘" 등 불용어만 있는 청크 제외
            if p_clean in ["날씨를", "날씨", "비교", "비교해줘", "알려줘", "그리고", "소개해줘", "검색해줘"]:
                continue
            # 3. 조사가 붙은 단독 키워드 처리 (예: "날씨는")
            if p_clean.endswith(("날씨를", "날씨는", "날씨가")):
                continue
                
            filtered_parts.append(p_clean)
            
        parts = filtered_parts # Update parts with filtered list
        
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
