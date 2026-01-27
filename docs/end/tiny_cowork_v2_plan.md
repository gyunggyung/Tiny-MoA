# 🤝 Tiny Cowork v2.0: Autonomous Multi-Agent System Plan

## 🎯 Goal
Tiny Cowork를 단순한 모아(MoA)의 확장을 넘어, 병렬 처리와 전문화된 에이전트(Workers)를 갖춘 **자율 협업 시스템**으로 진화시킵니다.

## ⚠️ Current Problems
- **Low Quality Summary**: 컨텍스트 주입 부족으로 인해 "내용이 없습니다"와 같은 부실한 요약 발생.
- **Sequential Execution**: 태스크를 하나씩 순서대로만 처리하여 지연 시간 발생.
- **UI Absence**: 현재 무엇이 진행 중인지 실시간으로 파악하기 어려움.

## 🚀 Proposed Changes

### 1. Parallel Orchestration (`src/tiny_moa/cowork/orchestrator_v2.py`)
- **ParallelTaskRunner**: `threading` 또는 `asyncio`를 사용하여 독립적인 태스크(예: 여러 파일 읽기, 검색)를 동시에 수행.
- **State Machine**: 태스크 간의 의존성(Dependencies)을 관리하여 병렬과 순차 처리를 혼합.

### 2. Specialized Workers (`src/tiny_moa/cowork/workers/`)
- **[Writer] Professional Writer**: 보고서, 메일, 문서 작성 전용 프롬프트와 Reasoner 모델을 사용하여 고품질 결과물 생성.
- **[Researcher] Deep Searcher**: RAG(로컬)와 Web Search(외부)를 결합하여 입체적인 정보 수집.
- **[Critic] Quality Guard**: 최종 결과물이 사용자의 의도에 맞는지 검토하고 필요시 수정을 요청하는 루프 추가.

### 3. Modern TUI Dashboard (`src/tiny_moa/ui/dashboard.py`)
- **Rich-based Interface**: 
    - **Task Board**: 현재 진행 중인 할 일 목록과 상태(병렬 진행 상황) 시각화.
    - **Agent Status**: 각 에이전트(Writer, Researcher 등)가 현재 무엇을 '생각'하는지 실시간 로그 출력.
    - **System Health**: 토큰 사용량 및 처리 시간 표시.

### 4. Skill Expansion (`src/tiny_moa/cowork/skills/`)
- **Browser Skill**: Playwright 또는 단순 검색을 넘어선 웹 페이지 파싱 능력 강화.
- **Formatter Skill**: JSON, CSV, Markdown, PDF(추후) 등 다양한 포맷으로의 전문적인 출력 지원.

## 🛠️ Execution Interface
- **Command Line**: `uv run python -m tiny_moa.main --interactive --cowork`
- **TUI Mode**: `uv run python -m tiny_moa.main --tui`

## ✅ Verification Plan

### Automated Tests
1. **Parallel Speed Test**: 3개 파일을 동시에 읽고 요약하는 속도가 기존 순차 방식보다 빠른지 확인.
2. **Quality Benchmark**: 이전의 부실했던 요약 결과와 v2.0의 결과물을 비교 분석.

### Manual Verification
1. **TUI Dashboard**: 실제 대화 중 대시보드가 실시간으로 업데이트되는지 확인.
2. **Crash Resilience**: 중간에 태스크가 실패해도 다른 병렬 태스크들이 영향을 받지 않는지 확인.
