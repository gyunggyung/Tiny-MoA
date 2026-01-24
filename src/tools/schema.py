"""
Tool Schema 정의
================
OpenAI Function Calling 형식과 호환되는 Tool 스키마
"""

from typing import Any

# Tool 정의 (OpenAI 형식 호환)
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g., 'Seoul', 'Tokyo', 'New York')"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for current information on any topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3, max: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get current date and time for a timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g., 'Asia/Seoul', 'UTC', 'America/New_York')"
                }
            },
            "required": []
        }
    },
    {
        "name": "search_news",
        "description": "Search recent news articles using DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "News search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5, max: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_wikipedia",
        "description": "Get Wikipedia article summary for a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic to search on Wikipedia"
                },
                "lang": {
                    "type": "string",
                    "description": "Language code (en, ko, ja, etc. Default: en)"
                }
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
                "url": {
                    "type": "string",
                    "description": "URL to read content from"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 2000)"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "execute_command",
        "description": "Execute a terminal/shell command (Windows/Linux). Use for running Python, checking system info, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute (e.g., 'python --version', 'dir', 'ls')"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)"
                }
            },
            "required": ["command"]
        }
    },
]


def get_tool_by_name(name: str) -> dict | None:
    """이름으로 Tool 스키마 조회"""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_tools_prompt() -> str:
    """LLM에 전달할 Tool 목록 프롬프트 생성"""
    tools_desc = []
    for tool in TOOLS:
        params = tool["parameters"]["properties"]
        required = tool["parameters"].get("required", [])
        
        param_strs = []
        for pname, pinfo in params.items():
            req_mark = "*" if pname in required else ""
            param_strs.append(f"  - {pname}{req_mark}: {pinfo['description']}")
        
        tools_desc.append(
            f"- {tool['name']}: {tool['description']}\n" + "\n".join(param_strs)
        )
    
    return "\n".join(tools_desc)


def validate_tool_call(name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    """Tool 호출 유효성 검사"""
    tool = get_tool_by_name(name)
    if not tool:
        return False, f"Unknown tool: {name}"
    
    required = tool["parameters"].get("required", [])
    for req in required:
        if req not in arguments:
            return False, f"Missing required parameter: {req}"
    
    return True, ""


if __name__ == "__main__":
    print("=== Tool Schema ===")
    print(get_tools_prompt())
    print("\n=== Validation Test ===")
    print(validate_tool_call("get_weather", {"location": "Seoul"}))
    print(validate_tool_call("get_weather", {}))
    print(validate_tool_call("unknown_tool", {}))
