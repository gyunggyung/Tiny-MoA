# Multi-Step Reasoning & Robust Orchestration Plan

## 🎯 Goal
Enable Tiny MoA to handle complex queries that require:
1.  **Temporal/Logical Decomposition**: Breaking "Compare X and Y" into "Get X", "Get Y", "Compare".
2.  **Dependency Management**: Using output of Step 1 as input for Step 2.
3.  **Deep Robustness**: Handling "soft errors" (e.g., API timeouts that return 200 OK) and persistent failures.

## 🏗️ Architecture Design

### 1. Decomposer Agent (The Planner)
Currently, `Brain` routes directly to a tool. We need a planning step for complex queries.
- **Input**: "Seoul weather compare yesterday vs today"
- **Output (Plan)**:
    1. `get_weather(location="Seoul", date="yesterday")`
    2. `get_weather(location="Seoul", date="today")`
    3. `synthesize(step1, step2)`

**Implementation Strategy:**
- Use a lightweight "Planner Prompt" in `Brain` or `Orchestrator`.
- If routing confidence is low for a single tool, fallback to Decomposition.

### 2. Multi-Step Execution Loop
Instead of a single `chat()` -> `tool()` -> `response()` flow, implement a loop:

```python
plan = brain.create_plan(user_input)
context = {}

for step in plan:
    # Resolve dependencies (e.g., use previous step's output)
    args = resolve_args(step.args, context)
    
    # Execute with Robust Retry
    result = execute_robust(step.tool, args)
    
    # Validation
    if is_soft_error(result):
        result = retry_or_ask_user(step)
        
    context[step.id] = result

final_response = brain.synthesize(context)
```

### 3. Semantic Error Detection (The "Timeout" Fix)
Currently, `wttr.in` returning "API timeout" is treated as `success: True` by the wrapper because the Python function ran without exception.
- **Fix**: Inspect `result` content for keywords: `timeout`, `error`, `failed`, `rate limit`.
- **Action**: Trigger the Retry Loop even if `success: True`.

### 4. Recursive Refinement
If a specific data point is missing (e.g., "Yesterday's weather is not available"), the Brain should:
1.  **Catch the missing data**.
2.  **Search for alternative sources** (e.g., fall back from `get_weather` to `search_web("Seoul weather yesterday history")`).
3.  **Resume** the comparison.

## 📅 implementation Roadmap

### Phase 2.1: Semantic Retry (Immediate)
- [ ] Update `orchestrator.py` to scan tool outputs for "soft errors".
- [ ] Trigger Retry Loop if "timeout" or "error" is found in the JSON result.

### Phase 2.2: Simple Decomposition
- [ ] Add `Brain.decompose()` method.
- [ ] If user asks "Compare", split into two parallel tool calls.
- [ ] Aggregate results in `integrate_response`.

### Phase 2.3: Full State Machine (Long Term)
- [ ] Isolate "Plan", "Execute", "Verify" into separate modules.
- [ ] Allow dynamic re-planning (if Step 1 fails, change Step 2).

## 📝 Example Flow (Future)

**User**: "서울 어제랑 오늘 날씨 비교해줘"

1.  **Decomposer**:
    - Task A: `get_weather("Seoul", date="current")`
    - Task B: `search_web("Seoul weather yesterday history")`
    - Task C: `compare(A, B)`

2.  **Executor**:
    - Run A -> Success (Temp: -1°C)
    - Run B -> Success (Temp: -5°C found in news)

3.  **Synthesizer**:
    - "Today is -1°C, which is warmer than yesterday's -5°C."
