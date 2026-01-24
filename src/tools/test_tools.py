import sys
import os

# 현재 디렉토리를 path에 추가하여 모듈 import 가능하게 함
sys.path.append(os.getcwd())

from src.tools.executor import ToolExecutor
import logging

# 로깅 설정
logging.basicConfig(
    filename='tool_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def run_test():
    executor = ToolExecutor()
    print("Test started... check tool_test.log")
    
    tests = [
        ("get_weather", {"location": "Seoul"}),
        ("search_web", {"query": "Python latest version", "num_results": 1}),
        ("search_news", {"query": "Artificial Intelligence", "num_results": 1}),
        ("search_wikipedia", {"query": "Albert Einstein", "lang": "en"}),
        ("read_url", {"url": "https://www.python.org", "max_chars": 500}),
        ("execute_command", {"command": "ver"}), # Windows
        ("calculate", {"expression": "2 + 3 * 4"}),
        ("get_current_time", {"timezone": "Asia/Seoul"})
    ]

    for name, args in tests:
        try:
            logging.info(f"Testing {name} with args: {args}")
            print(f"Running {name}...")
            result = executor.execute(name, args)
            logging.info(f"Result for {name}: {result}")
            print(f"Done {name}")
        except Exception as e:
            logging.error(f"Error testing {name}: {str(e)}")
            print(f"Error {name}")

if __name__ == "__main__":
    run_test()
