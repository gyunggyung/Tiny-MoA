# 🌐 Tiny MoA: Agent Ecosystem & Selection Vision

Tiny MoA를 단순한 고정형 시스템을 넘어, 사용자가 직접 에이전트를 선택하고 확장할 수 있는 **개방형 에이전트 생태계**로 진화시키기 위한 전략 문서입니다.

---

## 🛠️ 1. 사용자 정의 에이전트 선택 (User-Defined Selection)

미래의 Tiny MoA에서는 사용자가 모든 에이전트를 다 쓰는 대신, 특정 작업에 필요한 에이전트만 '구독'하거나 '지정'하여 실행할 수 있습니다.

### 📍 주요 기능
- **Agent White-listing**: 사용자가 `--agents researcher,writer`와 같이 명령어를 주면, 오케스트레이터는 지정된 두 에이전트의 능력 범위 내에서만 플랜을 생성합니다.
- **Dynamic Coupling**: 작업 시작 전, 현재 작업의 성격에 따라 사용자가 에이전트 리스트를 승인(Approve)하거나 수정하는 단계를 추가합니다.
- **Resource Optimization**: 선택되지 않은 에이전트 모델은 로드하지 않음으로써 VRAM 및 CPU 자원을 최적화합니다.

---

## 🏪 2. Tiny Agent Store (The Ecosystem)

GPTs나 앱스토어처럼, 유저들이 자신만의 전문화된 에이전트(Persona + Tools + Knowledge)를 만들고 공유할 수 있는 플랫폼으로 발전합니다.

### 🚀 에이전트 스토어 구성 요소
- **Custom Agent Builder**: 사용자가 특정 도구(Tool)와 프롬프트, 그리고 고유의 RAG 데이터셋을 결합하여 새로운 에이전트를 정의하는 인터페이스.
- **Standardized Agent Port**: 어떤 에이전트라도 Tiny MoA 오케스트레이터에 즉시 연동될 수 있도록 하는 표준 프로토콜(JSON 기반) 제공.
- **Marketplace Logic**: 유저들이 만든 전문 에이전트(예: '법률 분석 전문가', 'Next.js 코드 리뷰어', '한국어 감성 작가')를 검색하고 자신의 시스템에 추가하는 기능.

---

## 📈 3. 단계별 로드맵 (Roadmap)

### 1단계: 에이전트 인터페이스 표준화 (Current ~ Next)
- 모든 에이전트가 동일한 입력/출력 인터페이스를 갖도록 리팩토링.
- 설정 파일(`agents.yaml`)을 통해 에이전트 활성화/비활성화 기능 구현.

### 2단계: 사용자 선택 UI 구현
- TUI 또는 Web Dashboard에서 체크박스를 통해 참여 에이전트를 선택하는 기능.
- 런타임 중에 특정 에이전트에게 작업을 강제 할당하는 기능.

### 3단계: 외부 에이전트 임포트 (Agent Store Early-phase)
- 로컬 파일 형태의 커스텀 에이전트를 로드하는 기능.
- Hugging Face 또는 전용 저장소에서 에이전트 패키지를 다운로드하여 통합.

---
**비전 수립일:** 2026-01-26
**최종 목표:** 사용자가 직접 조립하는 AI 협업 엔진 🧩
