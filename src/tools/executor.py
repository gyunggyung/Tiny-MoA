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


def search_web(query: str, num_results: int = 5) -> dict[str, Any]:
    """
    DuckDuckGo 웹 검색 - API 키 불필요!
    """
    from duckduckgo_search import DDGS
    
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


def search_news(query: str, num_results: int = 5) -> dict[str, Any]:
    """
    DuckDuckGo 뉴스 검색
    """
    from duckduckgo_search import DDGS
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=num_results))
            return {
                "query": query,
                "num_results": len(results),
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
        return {"error": str(e), "query": query}


def search_wikipedia(query: str, lang: str = "en") -> dict[str, Any]:
    """
    Wikipedia 검색 - API 키 불필요!
    """
    import requests
    import urllib.parse
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
    
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "TinyMoA/1.0"})
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", ""),
                "extract": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "source": "wikipedia",
                "lang": lang
            }
        return {"error": f"Not found: {query}", "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e), "query": query}


def read_url(url: str, max_chars: int = 2000) -> dict[str, Any]:
    """
    URL 내용 읽기 - 웹페이지 텍스트 추출
    """
    import requests
    from html import unescape
    import re
    
    try:
        response = requests.get(
            url, 
            timeout=15, 
            headers={"User-Agent": "TinyMoA/1.0 (Web Reader)"}
        )
        response.raise_for_status()
        
        # HTML 태그 제거 (간단한 방식)
        text = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "url": url,
            "content": text[:max_chars],
            "total_length": len(text),
            "truncated": len(text) > max_chars,
            "source": "url_reader"
        }
    except Exception as e:
        return {"error": str(e), "url": url}


def execute_command(command: str, timeout: int = 30) -> dict[str, Any]:
    """
    터미널 명령어 실행 (Windows/Linux 호환)
    
    주의: 보안상 위험할 수 있음. 신뢰할 수 있는 명령만 실행.
    """
    import subprocess
    import platform
    
    # 위험한 명령어 차단
    dangerous_patterns = [
        "rm -rf", "del /s /q", "format", "mkfs",
        "shutdown", "reboot", "halt",
        "dd if=", "> /dev/",
        "chmod 777", "chmod -R",
        "curl | sh", "wget | sh",
    ]
    
    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in cmd_lower:
            return {
                "error": f"Blocked dangerous command pattern: {pattern}",
                "command": command
            }
    
    try:
        # 플랫폼에 따른 셸 설정
        if platform.system() == "Windows":
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        
        return {
            "command": command,
            "stdout": result.stdout[:5000] if result.stdout else "",
            "stderr": result.stderr[:1000] if result.stderr else "",
            "return_code": result.returncode,
            "success": result.returncode == 0,
            "platform": platform.system()
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s", "command": command}
    except Exception as e:
        return {"error": str(e), "command": command}


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
    from datetime import timezone as dt_timezone
    
    try:
        if timezone.upper() == "UTC":
            tz = dt_timezone.utc
        else:
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
        now = datetime.now(dt_timezone.utc)
        return {
            "timezone": "UTC (fallback)",
            "datetime": now.isoformat(),
            "formatted": now.strftime("%Y년 %m월 %d일 %H:%M:%S"),
            "error": f"Invalid timezone '{timezone}', using UTC. Error: {str(e)}"
        }


class ToolExecutor:
    """Tool 실행 관리자"""
    
    def __init__(self):
        # 이름 → 함수 매핑
        self.tools: dict[str, Callable] = {
            "get_weather": get_weather,
            "search_web": search_web,
            "search_news": search_news,
            "search_wikipedia": search_wikipedia,
            "read_url": read_url,
            "calculate": calculate,
            "get_current_time": get_current_time,
            "execute_command": execute_command,
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
    
    # 1. 날씨
    print("\n[1] 날씨 조회:")
    print(executor.execute("get_weather", {"location": "Seoul"}))
    
    # 2. 웹 검색 (DuckDuckGo)
    print("\n[2] 웹 검색 (Python):")
    # 실제 검색이 되므로 1개만 요청
    print(executor.execute("search_web", {"query": "Python latest version", "num_results": 1}))
    
    # 3. 뉴스 검색
    print("\n[3] 뉴스 검색 (AI):")
    print(executor.execute("search_news", {"query": "Artificial Intelligence", "num_results": 1}))
    
    # 4. 위키피디아
    print("\n[4] 위키피디아 (알베르트 아인슈타인):")
    print(executor.execute("search_wikipedia", {"query": "Albert Einstein", "lang": "en"}))
    
    # 5. URL 읽기 (python.org 예시)
    print("\n[5] URL 읽기 (python.org):")
    print(executor.execute("read_url", {"url": "https://www.python.org", "max_chars": 500}))

    # 6. 명령어 실행 (Windows 버전 확인)
    print("\n[6] 명령어 실행 (ver):")
    print(executor.execute("execute_command", {"command": "ver"})) # Windows version command
    
    # 7. 계산
    print("\n[7] 계산:")
    print(executor.execute("calculate", {"expression": "2 + 3 * 4"}))
    
    # 8. 현재 시간
    print("\n[8] 현재 시간:")
    print(executor.execute("get_current_time", {"timezone": "Asia/Seoul"}))
