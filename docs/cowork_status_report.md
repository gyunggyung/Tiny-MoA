# 🤝 Tiny Cowork 구현 현황 보고서

사용자의 고수준 목표를 자율적으로 분해하고 실행하는 **Tiny Cowork** 시스템의 핵심 프레임워크 구현이 완료되었습니다.

## ✅ 완료된 항목 (Implemented)

### 1. 🛡️ Safety Guard (`src/tiny_moa/cowork/safety.py`)
- **샌드박스 강제**: 모든 파일 접근이 지정된 폴더 내부에서만 일어나도록 검증.
- **위험 동작 감지**: `rm`, `delete`, `format` 등 파괴적인 키워드 감지 및 차단 로직.

### 2. 📂 Workspace Context (`src/tiny_moa/cowork/workspace.py`)
- **파일 관리 API**: 에이전트가 사용할 수 있는 안전한 `list_files`, `read_file`, `write_file` 메서드 제공.
- **컨텍스트 요약**: 현재 폴더 구조를 LLM이 이해하기 쉬운 텍스트 형식으로 변환.

### 3. 📋 Task Queue System (`src/tiny_moa/cowork/task_queue.py`)
- **태스크 구조화**: 각 작업의 상태(Pending, Running, Completed, Failed) 및 담당 에이전트 관리.
- **비동기 준비**: 향후 병렬 처리를 위한 큐 기반 아키텍처 구축.

### 4. 🧠 Planner Agent (`src/tiny_moa/cowork/planner.py`)
- **태스크 분해**: Brain(LFM-1.2B)을 사용하여 "폴더 정리해줘" 같은 모호한 명령을 구체적인 JSON 태스크 리스트로 변환.

### 5. ⚙️ Orchestrator Integration (`src/tiny_moa/orchestrator.py`)
- **`run_cowork_flow` 구현**: 
    1. 워크스페이스 로드 
    2. 플랜 생성 
    3. 태스크 순차 실행 (Tool, RAG, Brain 연동) 
    4. 최종 결과 통합 보고.

---

## 🚀 향후 과제 (Remaining Tasks)

### 1. 🧪 검증 (Verification)
- [ ] **샌드박스 테스트**: 폴더 외부 접근 시도 시 차단 여부 확인.
- [ ] **실제 시나리오 테스트**: "임시 폴더의 로그 파일들을 읽고 에러만 추려서 report.md로 만들어줘" 명령 수행 확인.

### 2. 📺 인터페이스 (UI)
- [x] **TUI (Terminal User Interface)**: Rich 기반 실시간 대시보드 및 로그 가시성 강화 완료.

### 3. 🛠️ 스킬 확장 (Skills)
- [x] **Web Search/News Skill**: DuckDuckGo를 이용한 실시간 정보 수집 강화 완료.
- [ ] **Complex Reasoner Skill**: 복합적인 논리 구조를 가진 질문에 대한 추론 능력 강화.

---

## ⚠️ 발견된 문제점 (V2.1 주요 과제)

### 1. 🧠 지능적 태스크 분해 (Intelligent Planning)
- **증상**: "한국 런던 도쿄 날씨 알려줘"와 같은 나열형 질문을 하나의 태스크로 묶어버리는 현상.
- **원인**: 플래너의 프롬프트가 '단순 질문은 1-2개 태스크로 제한'하도록 강제되어 있어 유연성 부족.
- **해결**: 리스트 형태의 요청을 감지하여 개별 태스크로 병렬화하는 능력 강화.

### 2. 🧩 컨텍스트 인지 (Context Awareness)
- **증상**: 도구 사용 시 필요한 인자를 누락하거나 잘못된 도구를 매핑하는 경우 발생.
- **해결**: `brain.py`의 `decompose_query` 휴리스틱 고도화 및 LLM 기반 분해 안정성 재확보.

---
**보고일:** 2026-01-26
**상태:** v2.0 Stable / v2.1 Planning & Testing 🧪

---

## 🗺️ v2.1 Intelligence Phase Roadmap
- **LLM-First**: Python 휴리스틱보다 모델의 추론력을 우선 활용.
- **Stress Testing**: LFM2.5의 지능적 한계를 측정하여 최적의 폴백 로직 구축.
- **Detailed Spec**: [v2_1_llm_intelligence_spec.md](file:///c:/github/MoA-PoC/docs/v2_1_llm_intelligence_spec.md) 참조.
