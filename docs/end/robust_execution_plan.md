# 🛡️ 시스템 안정성 및 명령 실행 개선 계획

## 📊 현상 분석

### 1. 성공 사례 (하지만 개선 필요)
> **입력**: "uv환경을 지금 프로젝트에 적용했는지 확인해봐."
- **흐름**: 
  1. Brain이 자연어 명령(`"Check the current..."`) 생성
  2. Orchestrator 방어 로직이 이를 감지하고 차단
  3. 키워드 폴백(`_infer_tool_from_keywords`)이 작동하여 `uv --version` 실행
  4. **성공**
- **시사점**: Orchestrator의 방어 로직("안전망")은 훌륭하게 작동했으나, Brain이 처음부터 올바른 명령을 내리지 못한 점은 아쉬움.

### 2. 실패 사례 (개선 시급)
> **입력**: "지금 몇 버전의 uv 환경을 쓰고 있는지 말해봐."
- **흐름**:
  1. Brain이 이를 `REASONER`(추론/코딩) 작업으로 오판
  2. Reasoner(Falcon-0.6B)가 도구 없이 내부 지식으로 답변 시도
  3. **환각(Hallucination)** 발생 (엉뚱한 Python 버전 언급)
- **시사점**: "버전", "환경" 등 확인이 필요한 질문은 반드시 `TOOL`로 라우팅되어야 함.

---

## 🛠️ 개선 방안

### 1단계: 라우팅 정밀도 향상 (Routing Precision)
Brain이 질문의 의도를 더 정확하게 파악하여 `REASONER`가 아닌 `TOOL`로 보내도록 훈련(프롬프트 강화)합니다.

- **Action**: `brain.py`의 라우팅 키워드 목록 확장.
  - "몇 버전", "환경", "쓰고 있는지", "말해봐" 등의 패턴이 보이면 `TOOL` 가중치 부여.
  - REASONER의 역할을 "순수 코딩 생성"과 "수학 풀이"로 더 좁게 정의.

### 2단계: 명령어 생성 최적화 (Command Generation)
Brain이 방어 로직에 걸리지 않는 "실행 가능한 셸 명령어"를 바로 생성하도록 합니다.

- **Action**: `Brain` 프롬프트에 `execute_command` 예시 추가.
  - *Bad*: "Check python version"
  - *Good*: "python --version"
  - 시스템 프롬프트에 "Do not explain, just generate command" 강조.

### 3단계: Reasoner의 도구 활용 (Tool-Enabled Reasoner)
Reasoner가 모르는 정보에 대해 환각을 일으키는 대신, 도구 사용을 요청하거나 모른다고 답하게 합니다.

- **Action**: Reasoner 프롬프트 수정.
  - "확실하지 않은 정보는 지어내지 말고, 정보가 부족하다고 말하라."
  - 장기적으로는 Reasoner도 `executor.py`를 알게 하여 필요시 도구 호출 가능하게 구조 변경 (MoA 아키텍처 심화).

### 4단계: 자기 수정 루프 (Self-Correction Loop)
도구 실행이 실패하거나 방어 로직에 걸렸을 때, 자동으로 재시도 시스템을 가동합니다.

- **Scenario**: 
  1. Brain: "Check uv" (Fail)
  2. System: "Error: Invalid command. Please provide exact shell command."
  3. Brain(Retry): "uv --version" (Success)

---

## 📅 실행 로드맵

1.  **즉시 적용**: `brain.py` 라우팅 키워드 및 프롬프트 추가 보정.
2.  **단기 목표**: `orchestrator.py`에 Brain 명령어 생성 실패 시 재시도(Retry) 로직 구현.
3.  **장기 목표**: Reasoner와 Tool의 통합.
