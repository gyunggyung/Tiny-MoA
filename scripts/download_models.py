"""
모델 다운로드 스크립트
=====================
huggingface-cli를 사용하여 GGUF 모델 다운로드
"""

import subprocess
import sys
from pathlib import Path

# 모델 정보
MODELS = {
    "brain": {
        "repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "filename": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        "description": "Brain (LFM2.5-1.2B-Instruct) - 라우터 & 한국어",
    },
    "brain-thinking": {
        "repo": "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        "filename": "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
        "description": "Brain Thinking (실험용)",
    },
    "reasoner": {
        "repo": "tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF",
        "filename": "Falcon-H1R-0.6B-Q4_0.gguf",  # 362MB
        "description": "Reasoner (Falcon-R-0.6B) - 코딩+수학",
    },
    "tool": {
        "repo": "tiiuae/Falcon-H1-Tiny-Tool-Calling-GGUF",
        "filename": "Falcon-H1-Tiny-Tool-Calling-90M-Q8_0.gguf",
        "description": "Tool Caller (선택적)",
    },
}


def download_model(model_key: str) -> bool:
    """단일 모델 다운로드"""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return False
    
    model = MODELS[model_key]
    print(f"\n📥 Downloading: {model['description']}")
    print(f"   Repo: {model['repo']}")
    print(f"   File: {model['filename']}")
    
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=model["repo"],
            filename=model["filename"]
        )
        print(f"✅ Downloaded: {model_key}")
        print(f"   Path: {path}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def download_all():
    """모든 필수 모델 다운로드"""
    required = ["brain", "reasoner"]
    
    print("🚀 Tiny MoA 모델 다운로드")
    print("=" * 50)
    
    for model_key in required:
        download_model(model_key)
    
    print("\n" + "=" * 50)
    print("✅ 필수 모델 다운로드 완료!")
    print("\n선택적 모델:")
    print("  - brain-thinking: python scripts/download_models.py brain-thinking")
    print("  - tool: python scripts/download_models.py tool")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tiny MoA 모델 다운로드")
    parser.add_argument(
        "models",
        nargs="*",
        help="다운로드할 모델 (없으면 필수 모델만)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="사용 가능한 모델 목록",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("사용 가능한 모델:")
        for key, info in MODELS.items():
            print(f"  {key}: {info['description']}")
    elif args.models:
        for model in args.models:
            download_model(model)
    else:
        download_all()
