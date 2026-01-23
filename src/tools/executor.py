"""
Tool Executor
=============
실제 API 호출 및 Tool 실행
"""

import json
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

# 개별 도구 함수들
def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    """
    날씨 정보 조회 (wttr.in API - 무료, API 키 불필요)
    """
    import requests
    
    try:
        # wttr.in API 호출 (JSON 형식)
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=30, headers={"User-Agent": "curl/7.0"})
        response.raise_for_status()
        data = response.json()
        
        current = data["current_condition"][0]
        
        # 온도 단위 처리
        if unit == "fahrenheit":
            temp = current["temp_F"]
            unit_symbol = "°F"
        else:
            temp = current["temp_C"]
            unit_symbol = "°C"
        
        return {
            "location": location,
            "temperature": f"{temp}{unit_symbol}",
            "condition": current["weatherDesc"][0]["value"],
            "humidity": f"{current['humidity']}%",
            "feels_like": f"{current['FeelsLikeC']}°C",
            "wind": f"{current['windspeedKmph']} km/h",
            "source": "wttr.in"
        }
    except requests.exceptions.Timeout:
        return {"error": "API timeout - please try again"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except (KeyError, IndexError) as e:
        return {"error": f"Invalid API response: {str(e)}"}


def search_web(query: str, num_results: int = 3) -> dict[str, Any]:
    """
    웹 검색 (Mock 또는 DuckDuckGo API)
    
    실제 구현 시: duckduckgo-search 패키지 사용
    """
    # Mock 데이터 (PoC용)
    # 실제 구현:
    # from duckduckgo_search import DDGS
    # with DDGS() as ddgs:
    #     results = list(ddgs.text(query, max_results=num_results))
    
    mock_results = [
        {
            "title": f"Search result for: {query}",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            "snippet": f"This is a mock search result for '{query}'. In production, this would be real search data."
        }
    ]
    
    return {
        "query": query,
        "num_results": min(num_results, len(mock_results)),
        "results": mock_results[:num_results],
        "source": "mock_data"
    }


def calculate(expression: str) -> dict[str, Any]:
    """
    수학 계산 (안전한 eval)
    """
    # 허용된 문자만 포함 확인 (보안)
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return {
            "expression": expression,
            "result": None,
            "error": "Invalid characters in expression. Only numbers and basic operators allowed."
        }
    
    try:
        # 안전한 eval (빌트인 함수 비활성화)
        result = eval(expression, {"__builtins__": {}}, {})
        return {
            "expression": expression,
            "result": result,
            "error": None
        }
    except Exception as e:
        return {
            "expression": expression,
            "result": None,
            "error": str(e)
        }


def get_current_time(timezone: str = "UTC") -> dict[str, Any]:
    """
    현재 시간 조회
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return {
            "timezone": timezone,
            "datetime": now.isoformat(),
            "formatted": now.strftime("%Y년 %m월 %d일 %H:%M:%S"),
            "error": None
        }
    except Exception as e:
        # 잘못된 타임존의 경우 UTC로 폴백
        now = datetime.now(ZoneInfo("UTC"))
        return {
            "timezone": "UTC (fallback)",
            "datetime": now.isoformat(),
            "formatted": now.strftime("%Y년 %m월 %d일 %H:%M:%S"),
            "error": f"Invalid timezone '{timezone}', using UTC"
        }


class ToolExecutor:
    """Tool 실행 관리자"""
    
    def __init__(self):
        # 이름 → 함수 매핑
        self.tools: dict[str, Callable] = {
            "get_weather": get_weather,
            "search_web": search_web,
            "calculate": calculate,
            "get_current_time": get_current_time,
        }
    
    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Tool 실행
        
        Args:
            tool_name: 도구 이름
            arguments: 도구 인자
            
        Returns:
            실행 결과 (dict)
        """
        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        try:
            result = self.tools[tool_name](**arguments)
            return {
                "success": True,
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "arguments": arguments,
                "error": str(e)
            }
    
    def execute_from_json(self, tool_call_json: str) -> dict[str, Any]:
        """
        JSON 문자열에서 Tool 호출 파싱 및 실행
        
        Args:
            tool_call_json: {"name": "tool_name", "arguments": {...}} 형식
        """
        try:
            call = json.loads(tool_call_json)
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})
            return self.execute(tool_name, arguments)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON: {e}"
            }


if __name__ == "__main__":
    executor = ToolExecutor()
    
    print("=== Tool Executor 테스트 ===\n")
    
    # 날씨
    print("1. 날씨 조회:")
    print(executor.execute("get_weather", {"location": "Seoul"}))
    print()
    
    # 검색
    print("2. 웹 검색:")
    print(executor.execute("search_web", {"query": "Python tutorial"}))
    print()
    
    # 계산
    print("3. 계산:")
    print(executor.execute("calculate", {"expression": "2 + 3 * 4"}))
    print()
    
    # 시간
    print("4. 현재 시간:")
    print(executor.execute("get_current_time", {"timezone": "Asia/Seoul"}))
    print()
    
    # JSON에서 실행
    print("5. JSON에서 실행:")
    print(executor.execute_from_json('{"name": "calculate", "arguments": {"expression": "100 / 4"}}'))
