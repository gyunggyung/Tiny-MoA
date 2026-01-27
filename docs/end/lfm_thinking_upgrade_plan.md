# 🧠 LFM2.5-1.2B-Thinking Model Upgrade Plan

## 1. Overview
Tiny Cowork v2.1의 목표인 '지능형 문맥 인지'와 '자율적 기획'을 달성하기 위해, 기존의 Base/Instruct 모델 대신 **LFM2.5-1.2B-Thinking** 모델을 도입합니다. 이 모델은 "사고 과정(Thinking Trace)"을 생성하여 복잡한 추론과 계획 수립 능력이 대폭 향상된 모델입니다.

## 2. Rationale (Why Thinking?)
- **High Reasoning Capability**: 1.2B 파라미터로 소형이지만, MATH-500 (88점), GPQA 등 추론 벤치마크에서 경쟁 모델을 압도함.
- **Agentic Suitability**: Tool Calling, 복합 질문 분해(Decomposition), RAG 정보 통합 등 '에이전트'업무에 최적화됨.
- **Efficient Edge Inference**: CPU/NPU 환경에서도 준수한 속도 (CPU 기준 약 200+ tok/s)를 제공하여 로컬 구동 가능.

## 3. Implementation Details

### Model Specs
- **ID**: `LiquidAI/LFM2.5-1.2B-Thinking-GGUF`
- **File**: `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` (~731MB)

### Recommended Parameters (Crucial)
Thinking 모델의 성능을 제대로 끌어내기 위해 제조사(LiquidAI) 권장 파라미터를 엄격히 준수해야 합니다.
- **Temperature**: `0.05` (매우 낮음) - 사고 과정의 일관성 유지
- **Top_K**: `50`
- **Repetition Penalty**: `1.05`

### Changes in `TinyMoA`
- **CLI Flag**: `--thinking` 옵션 추가 (기존 모델과 선택적 사용 가능)
- **Engine Logic**: `use_thinking=True`일 때 위 파라미터 셋을 적용하고, 모델 ID를 Thinking 버전으로 자동 전환.

## 4. Verification Plan
1. **Load Test**: `--thinking` 플래그로 실행 시 모델 자동 다운로드 및 로드 확인.
2. **Reasoning Test**:
   - Q: "서울과 도쿄의 날씨를 비교해줘" (Decomposition & Tool Calling 복합)
   - Q: "3.9와 3.11 중 어느 것이 더 큰가?" (Numeric Reasoning)
3. **Speed Benchmarking**: 실제로 로컬 환경에서 사고 과정이 포함된 응답 속도가 UX를 해치지 않는지 확인.
