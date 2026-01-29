# Current Issues and Performance Bottlenecks

## 1. Query Decomposition Logic (Critical)
The current heuristic decomposition is overly aggressive and produces "garbage" tasks by treating common verbs and adjectives as entities.

### Observed Behavior
**Query:** "앤트로픽과 삼성전자의 최신 뉴스들을 알려줘. 그리고 파악한 기사를 기반으로 최근 인공지능의 동향을 설명해줘."
**Resulting Tasks:**
- "앤트로픽 뉴스" (Correct)
- "삼성전자 뉴스" (Correct)
- "뉴스들 뉴스" (Incorrect - '뉴스들' is just 'news plural')
- "알려줘. 뉴스" (Incorrect - '알려줘' is 'tell me')
- "파악한 뉴스" (Incorrect - '파악한' is 'identified/grasped')
- "기반 뉴스" (Incorrect - '기반' is 'based on')
- "설명해줘. 뉴스" (Incorrect - '설명해줘' is 'explain')

**Query:** "지금 사용하고 있는 파이썬 버전과 uv 버전 알려줘"
**Resulting Tasks:**
- "사용" (Using)
- "있는" (Existing/ing)
- "파이썬" (Python)
- "버전" (Version)
- "uv"
**(All treated as separate tool lookups instead of a single conceptual query)**

### Cause
- The `decompose_query` function likely splits by spaces/particles but fails to identify or filter out common Korean predicates (verbs/adjectives) and auxiliary words when they don't match the hardcoded stopword list.
- The `stopwords` list is incomplete for the variety of Korean sentence endings.

## 2. System Performance (Bottleneck)
- **Symptom:** The user reports "speed is always the bottleneck."
- **Observation:**
    - Serial execution of tasks in some phases.
    - Model loading times (LFM2.5-1.2B) for every Brain invocation if not cached/kept alive properly.
    - Multiple "garbage" tasks (like "알려줘. 뉴스") execute real searches, wasting time and resources waiting for HTTP requests that yield irrelevant results.

## 3. Stability
- **Symptom:** "The command failed with exit code: 1" in Step 394.
- **Possible Cause:** Potential unhandled exception in the Tool logic or Brain routing when the decomposition result is malformed or empty.
- **Resource Warnings:** `ResourceWarning: unclosed file <_io.TextIOWrapper name='nul' ...>` appearing frequently.

## 4. TUI/UX Anomalies
- The user mentioned "behaving strangely" (`이상하게 동작하는 경우`).
- likely due to the decomposition logic flooding the task board with nonsensical tasks, making the agent look "confused" or "broken".
