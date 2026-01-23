# 🔧 Tool Calling 개선 계획 v2

> **목표:** 웹 검색을 포함한 실용적인 Tool Calling 시스템 구축

---

## 📋 현재 상태

### ✅ 구현 완료
| 도구 | 상태 | 데이터 소스 |
|------|------|-------------|
| `get_weather` | ✅ 작동 | wttr.in API (실제) |
| `get_current_time` | ✅ 작동 | 시스템 시간 |
| `calculate` | ✅ 작동 | Python eval |
| `search_web` | ⚠️ Mock | 가짜 데이터 |

### ❌ 해결 필요
1. **웹 검색이 Mock 데이터** → 실제 검색 결과 필요
2. **Falcon-90M 미사용** → 키워드 폴백에 의존 중
3. **JSON 파싱 오류 시 복구 부족**

---

## 🎯 개선 목표

```
우선순위:
1️⃣ 웹 검색 실제 작동 (DuckDuckGo) ← 가장 중요!
2️⃣ Falcon-90M 모델 활용
3️⃣ 새 도구 추가 (위키피디아, URL 읽기)
4️⃣ LFM2.5 JSON 검증/보정
```

---

## 🔍 Phase 1: 웹 검색 구현 (최우선)

### 옵션 비교

| 방법 | API 키 | 무료 | 안정성 | 추천 |
|------|--------|------|--------|------|
| **DuckDuckGo** | ❌ 불필요 | ✅ | ⭐⭐⭐ | ✅ 추천 |
| Google Custom Search | ✅ 필요 | 100회/일 | ⭐⭐⭐⭐ | |
| Bing Search | ✅ 필요 | 1000회/월 | ⭐⭐⭐⭐ | |
| SearXNG (셀프호스트) | ❌ | ✅ | ⭐⭐ | |

### DuckDuckGo 구현 (권장)

```python
# 설치
# pip install duckduckgo-search
# 또는
# uv add duckduckgo-search

from duckduckgo_search import DDGS

def search_web(query: str, num_results: int = 5) -> dict:
    """
    DuckDuckGo 웹 검색 - API 키 불필요!
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            
            return {
                "query": query,
                "num_results": len(results),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    for r in results
                ],
                "source": "duckduckgo"
            }
    except Exception as e:
        return {"error": str(e), "query": query}
```

### 테스트 코드

```python
# 테스트
result = search_web("Python tutorial", num_results=3)
for r in result["results"]:
    print(f"- {r['title']}: {r['url']}")
```

---

## 🦅 Phase 2: Falcon-90M 활용

### 현재 문제
- Falcon-90M 로드는 되지만 **실제로 사용되지 않음**
- `tool_hint`가 비어있어서 키워드 폴백으로 바로 넘어감

### 개선 방안

```python
def _handle_tool_call(self, user_input: str, tool_hint: str = "", verbose: bool = True):
    """개선된 Tool 호출 처리"""
    
    # 1단계: tool_hint가 있으면 바로 사용
    if tool_hint:
        tool_call = self._infer_tool_from_keywords(user_input, tool_hint)
    else:
        # 2단계: Falcon-90M으로 JSON 생성 시도
        if self.tool_caller and self.tool_caller._falcon:
            tool_call = self.tool_caller.generate_tool_call(user_input)
            
            # JSON 파싱 실패시 Brain으로 보정
            if "error" in tool_call and self.brain:
                tool_call = self._correct_with_brain(tool_call["raw"], user_input)
        else:
            # 3단계: 키워드 폴백
            tool_call = self._infer_tool_from_keywords(user_input, "")
    
    # 실행
    return self._execute_tool(tool_call)
```

### Falcon-90M 프롬프트 개선

```python
TOOL_CALLING_PROMPT = """<|im_start|>system
You are a function calling assistant. Given the user request, output ONLY a JSON object.

Available functions:
- search_web(query: str, num_results: int) - Search the web
- get_weather(location: str, unit: str) - Get weather info
- get_current_time(timezone: str) - Get current time
- calculate(expression: str) - Calculate math expression

Output format: {"name": "function_name", "arguments": {"param": "value"}}
<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
```

---

## 🛠️ Phase 3: 새 도구 추가

### 1. Wikipedia 검색

```python
def search_wikipedia(query: str, lang: str = "en") -> dict:
    """
    Wikipedia 검색 - API 키 불필요!
    """
    import requests
    
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{query}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", ""),
                "extract": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "source": "wikipedia"
            }
        return {"error": f"Not found: {query}"}
    except Exception as e:
        return {"error": str(e)}
```

### 2. URL 내용 읽기

```python
def read_url(url: str, max_chars: int = 2000) -> dict:
    """
    URL 내용 읽기 - 웹페이지 텍스트 추출
    """
    import requests
    from html import unescape
    import re
    
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "TinyMoA/1.0"})
        response.raise_for_status()
        
        # HTML 태그 제거 (간단한 방식)
        text = re.sub(r'<[^>]+>', ' ', response.text)
        text = unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "url": url,
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
            "source": "url_reader"
        }
    except Exception as e:
        return {"error": str(e), "url": url}
```

### 3. 뉴스 검색

```python
def search_news(query: str, num_results: int = 5) -> dict:
    """
    DuckDuckGo 뉴스 검색
    """
    from duckduckgo_search import DDGS
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=num_results))
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", "")
                    }
                    for r in results
                ],
                "source": "duckduckgo_news"
            }
    except Exception as e:
        return {"error": str(e)}
```

---

## 📋 업데이트된 Tool Schema

```python
TOOLS = [
    # 기존 도구
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {...}
    },
    {
        "name": "get_current_time",
        "description": "Get current time for a timezone",
        "parameters": {...}
    },
    {
        "name": "calculate",
        "description": "Calculate a math expression",
        "parameters": {...}
    },
    
    # 새 도구
    {
        "name": "search_web",
        "description": "Search the web using DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results (default: 5)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_news",
        "description": "Search recent news articles",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "News search query"},
                "num_results": {"type": "integer", "description": "Number of results (default: 5)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_wikipedia",
        "description": "Get Wikipedia article summary",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to search"},
                "lang": {"type": "string", "description": "Language code (en, ko, ja...)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_url",
        "description": "Read and extract text content from a URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to read"},
                "max_chars": {"type": "integer", "description": "Max characters to return"}
            },
            "required": ["url"]
        }
    }
]
```

---

## ⏱️ 구현 일정

| 단계 | 작업 | 예상 시간 | 의존성 |
|------|------|-----------|--------|
| 1 | DuckDuckGo 웹 검색 구현 | 30분 | `uv add duckduckgo-search` |
| 2 | Wikipedia 검색 구현 | 20분 | requests (이미 있음) |
| 3 | URL 읽기 구현 | 20분 | requests |
| 4 | 뉴스 검색 구현 | 15분 | duckduckgo-search |
| 5 | Tool Schema 업데이트 | 15분 | - |
| 6 | Falcon-90M 프롬프트 개선 | 30분 | - |
| 7 | LFM2.5 JSON 보정 로직 | 30분 | - |
| 8 | 키워드 폴백 확장 | 20분 | - |
| 9 | 통합 테스트 | 30분 | - |
| **총계** | | **~3시간** | |

---

## 🧪 테스트 시나리오

### 웹 검색 테스트
```bash
uv run python -m tiny_moa.main --query "Python 최신 버전 검색해줘"
# 예상: DuckDuckGo 검색 → 결과 3-5개 반환
```

### 뉴스 검색 테스트
```bash
uv run python -m tiny_moa.main --query "AI 관련 최신 뉴스"
# 예상: DuckDuckGo 뉴스 검색 → 뉴스 기사 반환
```

### Wikipedia 테스트
```bash
uv run python -m tiny_moa.main --query "인공지능이 뭐야?"
# 예상: Wikipedia 요약 반환
```

---

## ⚠️ 주의사항

1. **DuckDuckGo Rate Limit**: 너무 많은 요청 시 차단 가능
   - 해결: 요청 간 1-2초 딜레이 추가

2. **Falcon-90M 영어 전용**: 한국어 직접 처리 불가
   - 해결: 번역 파이프라인과 연동

3. **URL 읽기 보안**: 악성 URL 접근 가능
   - 해결: 도메인 화이트리스트 또는 사용자 확인

---

## 📦 필요 의존성

```bash
# 웹 검색 (필수)
uv add duckduckgo-search

# 선택적 확장
uv add beautifulsoup4  # 더 정교한 HTML 파싱
uv add newspaper3k     # 뉴스 기사 추출
```

---

## 🔮 향후 확장

| 도구 | 설명 | 우선순위 |
|------|------|----------|
| `translate` | 번역 도구 (현재는 파이프라인) | 중 |
| `summarize` | 긴 텍스트 요약 | 중 |
| `code_execute` | 코드 실행 (샌드박스) | 낮 |
| `file_read` | 로컬 파일 읽기 | 낮 |
| `image_describe` | 이미지 설명 (VLM) | 낮 |
