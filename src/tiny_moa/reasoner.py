"""
Reasoner 모델 래퍼 (Falcon-H1-Tiny-R-0.6B)
==========================================
- 코딩 작업 (Python, 알고리즘)
- 수학 문제 풀이 (AIME 스타일)
- LiveCodeBench 39% + MATH500 94%
"""

import os
from pathlib import Path
from typing import Optional
from llama_cpp import Llama

# Falcon-H1-Tiny-R 권장 파라미터 (반복 방지)
FALCON_R_PARAMS = {
    "temperature": 0.6,
    "repeat_penalty": 1.5,  # 반복 방지
    "top_p": 0.9,
}

# Reasoning 시스템 프롬프트 (간결하게)
REASONING_SYSTEM_PROMPT = """You are a coding and math assistant. Write clean Python code or solve math problems step by step."""


class Reasoner:
    """Falcon-H1-Tiny-R-0.6B 기반 Reasoner 모델 (코딩+수학)"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
    ):
        """
        Args:
            model_path: GGUF 모델 경로. None이면 기본 경로 사용
            n_ctx: 컨텍스트 길이
            n_threads: CPU 스레드 수. None이면 자동 감지
        """
        # 모델 경로 결정
        if model_path is None:
            # 1. 로컬 models/ 폴더 확인
            base_dir = Path(__file__).parent.parent.parent / "models" / "reasoner"
            gguf_files = list(base_dir.glob("*.gguf")) if base_dir.exists() else []
            
            if gguf_files:
                model_path = str(gguf_files[0])
            else:
                # 2. HuggingFace 캐시에서 자동 다운로드/찾기
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id="tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF",
                        filename="Falcon-H1R-0.6B-Q8_0.gguf"  # 공식 권장 Q8_0
                    )
                except Exception as e:
                    raise FileNotFoundError(
                        f"모델을 찾을 수 없습니다. 다운로드해주세요:\n"
                        f"huggingface-cli download tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF Falcon-H1-Tiny-R-0.6B-Q4_K_M.gguf\n"
                        f"Error: {e}"
                    )
        
        print(f"[Reasoner] Loading model from: {model_path}")
        
        # 스레드 수 결정
        if n_threads is None:
            n_threads = max(1, os.cpu_count() // 2)
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        
        print(f"[Reasoner] Loaded! (threads={n_threads}, ctx={n_ctx})")
    
    def solve(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        코딩 또는 수학 문제 풀이
        
        Args:
            prompt: 문제 설명
            max_tokens: 최대 생성 토큰 수
            
        Returns:
            풀이 결과
        """
        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            **FALCON_R_PARAMS,
        )
        
        return response["choices"][0]["message"]["content"]
    
    def code(self, task: str) -> str:
        """코딩 작업 전용 메서드"""
        prompt = f"Write Python code for the following task:\n\n{task}"
        return self.solve(prompt)
    
    def math(self, problem: str) -> str:
        """수학 문제 전용 메서드"""
        prompt = f"Solve the following math problem step by step:\n\n{problem}"
        return self.solve(prompt)


if __name__ == "__main__":
    # 테스트
    print("=== Reasoner 테스트 ===")
    reasoner = Reasoner()
    
    # 코딩 테스트
    print("\n[코딩 테스트]")
    code_result = reasoner.code("피보나치 수열의 n번째 항을 구하는 함수")
    print(code_result)
    
    # 수학 테스트
    print("\n[수학 테스트]")
    math_result = reasoner.math("1부터 100까지의 합을 구하시오.")
    print(math_result)
