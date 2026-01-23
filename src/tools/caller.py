"""
Tool Caller
===========
Falcon-H1-Tiny-Tool-Calling-90M + LFM2.5 검증 파이프라인
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from .schema import TOOLS, get_tools_prompt, validate_tool_call


class ToolCaller:
    """
    Tool Calling 전용 모델 래퍼
    
    전략:
    1. Falcon-90M: 빠른 JSON 생성 (영어 전용, 가끔 오타)
    2. LFM2.5 (Brain): JSON 검증 및 보정
    """
    
    def __init__(
        self,
        falcon_path: Optional[str] = None,
        brain_model = None,  # 이미 로드된 Brain 모델 재사용
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
    ):
        """
        Args:
            falcon_path: Falcon-90M GGUF 경로
            brain_model: 이미 로드된 Brain 모델 (검증용)
            n_ctx: 컨텍스트 길이
            n_threads: CPU 스레드 수
        """
        self.brain = brain_model  # 검증용 (선택적)
        self._falcon = None
        self.falcon_path = falcon_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, os.cpu_count() // 2)
    
    def _load_falcon(self):
        """Falcon-90M 모델 로드 (Lazy)"""
        if self._falcon is not None:
            return
        
        from llama_cpp import Llama
        
        # 모델 경로 결정
        if self.falcon_path:
            model_path = self.falcon_path
        else:
            # 로컬 models/ 폴더 확인
            base_dir = Path(__file__).parent.parent.parent / "models" / "tool_caller"
            gguf_files = list(base_dir.glob("*.gguf")) if base_dir.exists() else []
            
            if gguf_files:
                model_path = str(gguf_files[0])
            else:
                # HuggingFace에서 다운로드
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id="tiiuae/Falcon-H1-Tiny-Tool-Calling-90M-GGUF",
                        filename="Falcon-H1-Tiny-Tool-Calling-90M-Q8_0.gguf"
                    )
                except Exception as e:
                    raise FileNotFoundError(
                        f"Falcon-90M 모델을 찾을 수 없습니다:\n"
                        f"huggingface-cli download tiiuae/Falcon-H1-Tiny-Tool-Calling-90M-GGUF "
                        f"Falcon-H1-Tiny-Tool-Calling-90M-Q8_0.gguf\n"
                        f"Error: {e}"
                    )
        
        print(f"[ToolCaller] Loading Falcon-90M from: {model_path}")
        
        self._falcon = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        
        print(f"[ToolCaller] Loaded! (90M params, Q8_0)")
    
    @property
    def falcon(self):
        if self._falcon is None:
            self._load_falcon()
        return self._falcon
    
    def generate_tool_call(self, user_input: str) -> dict:
        """
        사용자 입력에서 Tool 호출 JSON 생성
        
        Returns:
            {"name": "tool_name", "arguments": {...}} 또는
            {"error": "..."} 실패 시
        """
        # Falcon-90M용 프롬프트
        tools_desc = get_tools_prompt()
        
        prompt = f"""<|im_start|>system
You are a function calling AI. Given the user's request, respond with a JSON object to call the appropriate function.

Available functions:
{tools_desc}

Respond ONLY with a valid JSON object in this format:
{{"name": "function_name", "arguments": {{"param": "value"}}}}
<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        
        output = self.falcon(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>", "\n\n"],
            temperature=0.1,  # 낮은 temperature로 안정적인 JSON 생성
            top_p=0.9,
            echo=False
        )
        
        content = output["choices"][0]["text"].strip()
        
        # JSON 추출 시도
        try:
            # JSON 블록 추출
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                
                # 유효성 검사
                if "name" in result:
                    valid, error = validate_tool_call(
                        result["name"],
                        result.get("arguments", {})
                    )
                    if valid:
                        return result
                    else:
                        return {"error": error, "raw": content}
            
            return {"error": "No valid JSON found", "raw": content}
            
        except json.JSONDecodeError as e:
            # LFM2.5로 보정 시도 (Brain이 있는 경우)
            if self.brain:
                return self._correct_with_brain(content, user_input)
            return {"error": f"JSON parse error: {e}", "raw": content}
    
    def _correct_with_brain(self, broken_json: str, original_request: str) -> dict:
        """
        LFM2.5 (Brain)로 깨진 JSON 보정
        """
        if not self.brain:
            return {"error": "No Brain model for correction", "raw": broken_json}
        
        correction_prompt = f"""Fix this broken JSON for a function call.
Original request: "{original_request}"
Broken JSON: {broken_json}

Available functions: get_weather, search_web, calculate, get_current_time

Respond with ONLY a valid JSON object like:
{{"name": "function_name", "arguments": {{"param": "value"}}}}"""
        
        try:
            corrected = self.brain.direct_respond(
                correction_prompt,
                system_prompt="You are a JSON repair assistant. Output only valid JSON, nothing else."
            )
            
            # JSON 추출
            start = corrected.find("{")
            end = corrected.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(corrected[start:end])
                if "name" in result:
                    return result
            
            return {"error": "Brain correction failed", "raw": corrected}
            
        except Exception as e:
            return {"error": f"Brain correction error: {e}", "raw": broken_json}
    
    def needs_tool(self, user_input: str) -> bool:
        """
        Tool 호출이 필요한지 빠르게 판단 (키워드 기반)
        
        나중에 Brain의 라우팅으로 대체 가능
        """
        tool_keywords = {
            "get_weather": ["날씨", "weather", "기온", "온도", "temperature", "비", "눈", "맑음"],
            "search_web": ["검색", "search", "찾아", "알려줘", "뭐야", "누구", "어디", "최신", "뉴스"],
            "calculate": ["계산", "calculate", "더하기", "빼기", "곱하기", "나누기", "+", "-", "*", "/", "="],
            "get_current_time": ["시간", "time", "몇시", "날짜", "date", "오늘"],
        }
        
        user_lower = user_input.lower()
        
        for tool_name, keywords in tool_keywords.items():
            if any(kw in user_lower for kw in keywords):
                return True
        
        return False


if __name__ == "__main__":
    print("=== Tool Caller 테스트 ===")
    print("(Falcon-90M 모델 필요)")
    
    # 키워드 테스트만 (모델 없이)
    caller = ToolCaller.__new__(ToolCaller)
    caller.brain = None
    caller._falcon = None
    
    test_inputs = [
        "서울 날씨 어때?",
        "Python 검색해줘",
        "1+2*3 계산해",
        "지금 몇시야?",
        "안녕하세요!",  # Tool 불필요
    ]
    
    print("\n키워드 기반 Tool 필요 여부:")
    for inp in test_inputs:
        needs = caller.needs_tool(inp)
        print(f"  '{inp}' → Tool 필요: {needs}")
