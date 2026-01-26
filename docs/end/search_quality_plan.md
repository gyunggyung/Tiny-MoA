# 🔍 검색 품질 및 결과 관련성 개선 계획

## 🚨 문제점

현재 `search_web` 도구는 사용자의 자연어 입력을 거의 그대로 검색 엔진(DuckDuckGo)에 전달하고 있습니다. 이로 인해 다음과 같은 문제가 발생합니다:

1.  **불필요한 문장 포함**: "아인슈타인에 대해서 검색해서..."와 같은 문장이 검색어에 포함되어 검색 결과의 정확도를 떨어뜨림.
2.  **검색 엔진의 한계**: DuckDuckGo가 복합적인 한국어 쿼리를 처리하는 데 한계가 있을 수 있음(엉뚱한 기술 문서나 코드 관련 결과 반환).
3.  **검증 부재**: 반환된 검색 결과가 사용자의 질문과 실제로 관련이 있는지 확인하는 절차가 없음.

---

## 🛠️ 개선 방안

### 1단계: 쿼리 최적화 (Query Optimization) [우선순위: 높음]

Brain 모델(LFM2.5)이 Tool 호출 시 **검색 엔진 친화적인 키워드**를 생성하도록 유도합니다.

- **현재**: `search_web({"query": "아인슈타인에 대해서 검색해서 최신 정보를"})`
- **개선**: `search_web({"query": "Albert Einstein biography latest news"})` 또는 `search_web({"query": "아인슈타인 최신 연구 뉴스"})`
- **실행**: `brain.py`나 `tool_caller`의 프롬프트에 "Search engine optimized keywords only" 지시 추가.

### 2단계: 결과 리랭킹 (Reranking & Filtering) [우선순위: 중간]

검색된 결과(Title, Snippet)가 사용자의 원래 질문과 관련이 있는지 로컬에서 빠르게 판단하여 노이즈를 제거합니다.

- **방법**:
    - `sentence-transformers` 같은 경량 임베딩 모델 사용 (옵션).
    - 또는 Brain 모델에게 "이 결과가 질문과 관련이 있는가?"라고 한번 더 물어보고(Self-Reflection) 필터링. e.g., "ICD-10 Code" 결과는 아인슈타인 질문과 무관하므로 제거.

### 3단계: 쿼리 확장 및 병렬 검색 (Query Expansion) [우선순위: 낮음]

하나의 질문을 여러 개의 검색어로 변환하여 다양한 각도에서 정보를 수집합니다.

- 예: "아인슈타인" -> ["Albert Einstein physics", "General Relativity", "Einstein biography"]
- 3개의 쿼리로 병렬 검색 후 결과 통합.

### 4단계: 검색 엔진 다변화 [우선순위: 낮음]

- DuckDuckGo 외에 Google Custom Search (API Key 필요하지만 정확도 높음) 또는 Bing Search 옵션 추가 고려.
- 또는 `search_wikipedia`를 우선적으로 활용하도록 라우팅 전략 수정.

---

## 📅 실행 계획 예시

1.  **Phase 1 (즉시 가능)**: Brain/Falcon 프롬프트 수정하여 **"검색어 정제"** 로직 추가.
2.  **Phase 2**: `executor.py`에 간단한 키워드 매칭 필터링 추가 (질문 키워드가 결과에 포함되었는지).
3.  **Phase 3**: Reranking 모델 도입 검토.
