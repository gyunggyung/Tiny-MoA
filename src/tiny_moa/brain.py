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
- REASONER: STRICTLY for coding tasks (Python implementation) and complex math problems only. Do NOT use for search or general questions.
- TOOL: For ANY requests requiring external information (weather, news, definitions), checking system status, verify commands, or real-time data.
- DIRECT: For general conversation, greetings, translations, and internal knowledge.

Respond with a JSON object:
{"route": "REASONER" or "TOOL" or "DIRECT", "specialist_prompt": "optimized search keywords for TOOL. For 'execute_command', provide the EXACT shell command (e.g., 'uv --version'). Do NOT provide descriptions.", "tool_hint": "tool name if TOOL route"}

Examples:
- "피보나치 함수 작성해줘" → {"route": "REASONER", "specialist_prompt": "Write a Python function for Fibonacci sequence", "tool_hint": ""}
- "서울 날씨 어때?" → {"route": "TOOL", "specialist_prompt": "Seoul", "tool_hint": "get_weather"}
- "아인슈타인 최신 정보" → {"route": "TOOL", "specialist_prompt": "Albert Einstein latest news", "tool_hint": "search_news"}
- "uv가 뭐야?" → {"route": "TOOL", "specialist_prompt": "what is uv python tool", "tool_hint": "search_web"}
- "지금 프로젝트에 uv 적용됐는지 확인해봐" → {"route": "TOOL", "specialist_prompt": "uv --version", "tool_hint": "execute_command"}
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
        
        print(f"[Brain] Loading model from: {model_path}")
        
        # 스레드 수 결정 (CPU 코어의 절반 권장)
        if n_threads is None:
            n_threads = max(1, os.cpu_count() // 2)
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        
        print(f"[Brain] Loaded! (threads={n_threads}, ctx={n_ctx})")
    
    def route(self, user_input: str) -> dict:
        """
        사용자 입력을 분석하여 라우팅 결정
        
        Returns:
            {"route": "REASONER" | "DIRECT", "specialist_prompt": str}
        """
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
        
        # 파싱 실패 시 키워드 기반 폴백
        user_lower = user_input.lower()
        
        # TOOL 키워드 우선 체크 (외부 정보 필요)
        keywords_tool = {
            "get_weather": ["날씨", "weather", "기온", "온도", "temperature"],
            "search_web": ["검색", "search", "찾아봐", "알려줘", "뭐야", "누구", "최신", "search_web", "news", "뉴스"],
            "get_current_time": ["시간", "time", "몇시", "날짜", "date", "오늘"],
            "execute_command": ["확인", "check", "verify", "run", "version", "버전", "실행", "command", "ls", "dir"],
        }
        for tool_name, keywords in keywords_tool.items():
            if any(kw in user_lower for kw in keywords):
                return {"route": "TOOL", "specialist_prompt": "", "tool_hint": tool_name}
        
        # REASONER 키워드
        keywords_reasoner = ["코드", "함수", "구현", "python", "알고리즘", "수학", "증명", "aime", "code", "function", "fibonacci"]
        if any(kw in user_lower for kw in keywords_reasoner):
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
        sys_content = system_prompt or "You are a helpful assistant."
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
        system_prompt = """You are a helpful assistant. 
The user asked a question and a specialist provided the following answer.
Present this answer clearly to the user in their language (Korean if they asked in Korean).
Do not add unnecessary explanations, just format the answer nicely."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {user_input}\n\nSpecialist answer:\n{specialist_output}"},
        ]
        
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            **self.params,
        )
        
        return response["choices"][0]["message"]["content"]


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
