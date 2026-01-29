# 라우팅 및 태스크 분해 기능 테스트 계획

> 작성일: 2026-01-29  
> 목적: `Brain.route()` 및 `Brain.decompose_query()` 기능의 정확도 검증

---

## 1. 배경 및 목적

### 1.1 테스트 대상
| 기능 | 위치 | 설명 |
|------|------|------|
| **라우팅 (Routing)** | `Brain.route()` | 사용자 입력을 분석하여 `TOOL`, `REASONER`, `DIRECT` 중 적절한 경로 결정 |
| **태스크 분해 (Decomposition)** | `Brain.decompose_query()` | 복잡한 질문을 여러 개의 간단한 쿼리로 분해 |

### 1.2 목표
1. **라우팅**: 질문 유형에 따라 올바른 경로(`TOOL`/`REASONER`/`DIRECT`)로 분기되는지 확인
2. **태스크 분해**: 복잡한 비교 쿼리가 적절히 분해되는지 확인
3. **LoRA 학습 데이터**: 실패 케이스 수집 → 향후 Fine-tuning 학습 데이터로 활용

---

## 2. 테스트 폴더 구조

```
c:\github\MoA-PoC\
├── src/
│   └── function_tests/          # [NEW] 기능 테스트 모듈
│       ├── __init__.py
│       ├── test_routing.py      # 라우팅 테스트
│       ├── test_decomposition.py # 태스크 분해 테스트
│       └── test_cases.py        # 테스트 케이스 정의
```

> **폴더 위치 결정**: `src/` 아래에 배치하여 모듈 import가 용이하도록 함

---

## 3. 테스트 케이스

### 3.1 라우팅 테스트 (`test_routing.py`)

#### 3.1.1 TOOL 라우팅 (날씨, 검색 등)

| 입력 쿼리 | 기대 결과 | 비고 |
|-----------|-----------|------|
| `서울 날씨 알려줘` | `TOOL` + `get_weather` | 단순 날씨 조회 |
| `오늘 서울 뉴스` | `TOOL` + `search_news` | 뉴스 검색 |
| `python 버전 확인해줘` | `TOOL` + `execute_command` | 명령어 실행 |

#### 3.1.2 REASONER 라우팅 (코딩, 추론)

| 입력 쿼리 | 기대 결과 | 비고 |
|-----------|-----------|------|
| `피보나치 함수 작성해줘` | `REASONER` | 코드 생성 |
| `AIME 2024 문제 풀어봐` | `REASONER` | 수학 추론 |

#### 3.1.3 DIRECT 라우팅 (일반 대화)

| 입력 쿼리 | 기대 결과 | 비고 |
|-----------|-----------|------|
| `안녕하세요!` | `DIRECT` | 인사 |
| `이 내용 요약해줘` | `DIRECT` | 요약 요청 |

---

### 3.2 태스크 분해 테스트 (`test_decomposition.py`)

#### 3.2.1 단순 쿼리 (분해 불필요)

| 입력 쿼리 | 기대 결과 | 비고 |
|-----------|-----------|------|
| `서울 날씨 알려줘` | `["서울 날씨 알려줘"]` 또는 `["서울 날씨"]` | 1개 반환 |
| `서울과 부산 날씨 비교해봐` | `["서울 날씨", "부산 날씨"]` | 2개 분해 |

#### 3.2.2 복잡 쿼리 (분해 필요)

| 입력 쿼리 | 기대 결과 | 비고 |
|-----------|-----------|------|
| `서울, 대전, 그리고 도쿄의 날씨와 런던 날씨를 비교해봐` | `["서울 날씨", "대전 날씨", "도쿄 날씨", "런던 날씨"]` | 4개 도시 + 비교 태스크 |
| `서울이랑 도쿄 날씨 알려줘` | `["서울 날씨", "도쿄 날씨"]` | 조사 `이랑` 처리 |

#### 3.2.3 비교 태스크 분해

> **핵심**: 날씨를 찾은 후 "비교하기" 태스크도 별도로 분해해야 함

| 입력 쿼리 | 기대 분해 결과 |
|-----------|----------------|
| `서울과 부산 날씨 비교해봐` | 1. `서울 날씨` 조회<br>2. `부산 날씨` 조회<br>3. 비교 분석 (DIRECT) |

> ⚠️ **현재 문제점**: 비교 분석 태스크가 별도로 분리되지 않음 → LoRA 학습 대상

---

## 4. 실행 계획

### 4.1 Phase 1: 라우팅 테스트 (빠른 실행, ~1분)

```powershell
cd c:\github\MoA-PoC
python -m src.function_tests.test_routing
```

- LLM 호출 없이 Fast Path 키워드 매칭 테스트
- 결과: JSON 형태로 성공/실패 로그 저장

### 4.2 Phase 2: 태스크 분해 테스트 (~3분)

```powershell
cd c:\github\MoA-PoC
python -m src.function_tests.test_decomposition
```

- LLM 호출 포함 (Brain 모델 필요)
- 결과: 분해 결과와 기대값 비교 → 정확도 리포트

### 4.3 Phase 3: 실패 케이스 분석

```powershell
cd c:\github\MoA-PoC
python -m src.function_tests.analyze_failures
```

- 실패한 케이스를 JSONL 형태로 저장
- LoRA 학습용 데이터셋 포맷으로 변환 가능

---

## 5. 성공 기준

| 기능 | 목표 정확도 | 비고 |
|------|-------------|------|
| 라우팅 | ≥ 95% | Fast Path + LLM 결합 |
| 태스크 분해 (단순) | ≥ 90% | 1-2개 도시 |
| 태스크 분해 (복잡) | ≥ 70% | 3개 이상 + 비교 태스크 |

---

## 6. 향후 계획

1. **실패 케이스 수집**: 테스트 결과에서 실패한 케이스를 `data/failures.jsonl`에 저장
2. **LoRA Fine-tuning**: 수집된 데이터로 `decompose_query` 전용 어댑터 학습
3. **A/B 테스트**: 기존 모델 vs LoRA 적용 모델 비교

---

## 7. 참고 파일

- [brain.py](file:///c:/github/MoA-PoC/src/tiny_moa/brain.py) - `route()`, `decompose_query()` 구현
- [verify_decomposition.py](file:///c:/github/MoA-PoC/tests/verify_decomposition.py) - 기존 분해 테스트
- [orchestrator.py](file:///c:/github/MoA-PoC/src/tiny_moa/orchestrator.py) - 전체 파이프라인
