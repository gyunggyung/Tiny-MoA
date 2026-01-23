"""
Tiny MoA CLI 진입점
==================
python -m tiny_moa.main [--interactive]
"""

import argparse
from tiny_moa.orchestrator import TinyMoA, interactive_mode
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Tiny MoA - GPU Poor를 위한 AI 군단",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python -m tiny_moa.main                    # 기본 테스트 실행
  python -m tiny_moa.main --interactive      # 대화형 모드
  python -m tiny_moa.main --query "피보나치 함수 작성해줘"
        """,
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드 실행",
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="단일 쿼리 실행",
    )
    
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="LFM Thinking 모델 사용 (실험 중)",
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        moa = TinyMoA(use_thinking=args.thinking)
        moa.chat(args.query)
    else:
        # 기본 테스트
        console.print("[bold]🧪 Tiny MoA 기본 테스트[/bold]\n")
        
        moa = TinyMoA(use_thinking=args.thinking)
        
        test_queries = [
            "안녕하세요! 반갑습니다.",
            "피보나치 수열의 10번째 항을 구하는 Python 함수를 작성해줘.",
            "1부터 100까지의 합은?",
        ]
        
        for query in test_queries:
            console.print(f"\n{'='*60}")
            moa.chat(query)


if __name__ == "__main__":
    main()
