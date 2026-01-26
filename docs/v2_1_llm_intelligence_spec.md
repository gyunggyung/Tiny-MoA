# 🧠 Tiny Cowork v2.1: LLM-Driven Intelligence Spec

## 🎯 Vision: "Beyond Heuristics"
v2.0까지는 정규식(Regex)과 단순 키워드 매핑을 통해 안정성을 확보했습니다. v2.1에서는 **LFM2.5 (1.2B) 모델의 추론 능력**을 신뢰하고, Python 코드의 개입을 최소화하여 인공지능 비서 본연의 '지능적 기획력'을 극대화합니다.

## 🚀 Key Strategy: LLM-First Planning

### 1. 지능형 쿼리 분해 (LLM Decomposition)
- **Problem**: "A B C 날씨"를 하나의 태스크로 묶어버리는 현상.
- **Solution**: `brain.py`에서 Python의 개입(정규식)을 줄이고, LLM이 문장에서 독립적인 '실행 단위(Entity)'를 직접 추출하도록 프롬프트를 고도화합니다.
- **Goal**: 공백, 특수문자, 복합 접속사가 섞인 불완전한 문장에서도 병렬화 가능한 요소를 100% 식별.

### 2. 자율 플래너 고도화 (Autonomous Planner)
- **Contextual Mapping**: 사용자의 목표뿐만 아니라 워크스페이스 내 파일들의 성격(로그, 문서, 코드)을 분석하여 최적의 에이전트(`tool`, `rag`, `brain`)를 동적으로 선택.
- **Infinite Scaling**: 태스크 개수의 제한을 풀고, 모델이 판단하는 최적의 세분화 단계에 따라 n개의 작업을 자동 생성.

## 🧪 LLM Intelligence Testing (Stress Test)
LFM2.5 모델이 실제로 얼마나 복잡한 명령까지 견디는지, 그리고 그 한계점은 어디인지 검증합니다.

| 테스트 유형 | 시나리오 | 기대 결과 |
| :--- | :--- | :--- |
| **단순 나열** | `서울 도쿄 런던 베를린 파리 뉴욕 날씨 확인` | 6개 독립 `tool` 태스크 생성 |
| **도메인 혼합** | `docs/log.txt 읽고 에러 분석한 다음 오늘 날씨랑 섞어서 보고서 써` | `rag` → `brain` → `tool` 순서의 논리적 체인 형성 |
| **불완전 문장** | `파이썬 버전... 그리고 뉴스... 딥마인드` | `execute_command`와 `search_news`로 자동 보정 |
| **극한 상황** | 10개 이상의 병렬 태스크 강제 유도 | `llama_decode` 안정성 및 TUI 렌더링 부하 측정 |

## ⚠️ Fallback & Safety
- **Stability Guard**: LLM이 JSON 형식을 깨뜨리거나 응답에 실패할 경우, 프로젝트의 기반이 되는 Python 휴리스틱(정규식)이 즉시 개입하여 최소한의 실행 가능성(Minimal Execution)을 보장합니다.

---
**보고일:** 2026-01-26
**상태:** Planning / Experimental 🧪
