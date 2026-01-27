# 🤝 Tiny Cowork v2.1 Unified Master Roadmap

이 문서는 Tiny Cowork의 현재 구현 상태와 **v2.1 'Intelligent Context Awareness'**를 위한 지능 고도화 전략을 한곳에 집대성한 통합 마스터 플랜입니다.

---

## 🟢 1. 현재 구현 현황 (Current Status)

v2.0을 통해 병렬 처리와 TUI 가시성을 갖춘 핵심 프레임워크가 안착되었습니다.

### ✅ 완료된 구성 요소
- **Intelligent Decomposition**: Thinking Model 기반의 복합 질문 분해 및 불용어 필터링 구현.
- **English-First Strategy**: 영어로 추론/생성 후 한국어로 번역하는 전략으로 속도와 품질 동시 확보.
- **Context Awareness**: `--n-ctx` 옵션을 통한 긴 문맥(12k+) 지원으로 Thinking 과정의 잘림 방지.
- **Robust Tooling**: 날씨 도구의 입력 보정(Sanitization) 및 도시명 매핑으로 정확도 향상.
- **Parallel Orchestrator**: 태스크 간 의존성을 관리하며 병렬 실행 지원.
- **Modern TUI**: Rich 기반의 실시간 태스크 보드 및 상세 액티비티 로그 창.

### ⚠️ 현재의 한계 (Pain Points / Resolved)
- **[해결됨] 나열형 질문 처리**: 모델 기반 분해로 "서울 도쿄 날씨" 등 복합 질문 정확히 처리.
- **[해결됨] 휴리스틱 의존**: LLM 추론을 우선하고 정규식은 보조 수단으로 전환.
- **[개선중] 플래너 제약**: 프롬프트 단순화 및 영어 지시로 Reasoning 효율 극대화.

---

## 🧠 2. v2.1 지능 고도화 전략 (LLM-First Strategy)
2026-01-27 기준, Thinking Model 도입 및 주요 난제 해결 완료.

### 🚀 핵심 구현 방향 (Status Update)
1. **모델 기반 쿼리 분해 (Completed)**
   - LFM Thinking 모델이 질문을 분석하여 독립적인 Tool Task로 분해 성공.
   - "날씨를" 같은 조사 찌꺼기를 필터링하는 후처리 로직 적용.
2. **English-First Reporting (New)**
   - Thinking Model의 사고 효율을 위해 내부 처리는 영어로 수행.
   - 최종 단계에서 Translation Pipeline이 한국어로 변환하여 사용자 경험 유지.
3. **지능적 폴백 (Hybrid Safety)**
   - LLM 분석 실패 시에만 정규식이 개입 (기존 유지).

### 🧪 스트레스 테스트 결과
- **복합 비교**: "서울과 도쿄, 런던 그리고 광주의 날씨 비교" -> 4개 도시 분해 및 정확한 수치 리포트 성공.
- **속도 최적화**: 프롬프트 단순화로 리포트 생성 시간 60% 이상 단축.

---

## 🔭 3. 향후 확장 비전 (Future Vision)

- **Web Dashboard (GUI)**: React/Next.js 기반의 현대적이고 미려한 웹 인터페이스.
- **Interactive Shell**: 이전 대화 맥락(Context)을 완벽히 기억하는 연속 대화 모드.
- **Vision/Coding Specialist**: 멀티모달 분석 및 정밀 코드 작성이 가능한 전용 에이전트 추가.
- **Web RAG**: 검색된 URL의 본문 내용을 자동으로 추출하고 분석하는 Deep Searcher 강화.
- **Agent Ecosystem**: [에이전트 스토어](file:///c:/github/MoA-PoC/docs/agent_ecosystem_vision.md) 생태계 구축.
- **All-in-One GUI App**: [독립형 애플리케이션](file:///c:/github/MoA-PoC/docs/tiny_cowork_app_vision.md)으로 진화.

---
**최종 업데이트:** 2026-01-27
**상태:** v2.1 Core Logic Implemented & Verified ✅
