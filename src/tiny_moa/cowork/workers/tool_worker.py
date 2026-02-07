from src.tiny_moa.cowork.workers.base import BaseWorker

class ToolWorker(BaseWorker):
    def __init__(self, name: str, logger, orchestrator):
        super().__init__(name, logger)
        self.orchestrator = orchestrator

    def execute(self, task_description: str, **kwargs) -> str:
        self.logger.info(f"[{self.name}] Tool processing: {task_description}")
        try:
            # [Fix] Brain 모델 병렬 충돌 방지: orchestrator.chat() 대신 직접 tool 실행
            # orchestrator.chat()은 Brain 모델을 사용하므로 병렬 실행 시 llama_decode 오류 발생
            
            # 1. Tool hint 추론 (간단한 키워드 기반)
            task_lower = task_description.lower()
            
            # [Fix] 명령어 실행 키워드 감지 (버전 확인, 파일 목록 등)
            # "uv 버전", "파이썬 버전", "파일 목록" 등
            if any(k in task_lower for k in ["version", "버전", "-v", "--version"]):
                # 버전 확인 명령어 추론
                has_uv = "uv" in task_lower
                has_python = "python" in task_lower or "파이썬" in task_lower
                
                # 둘 다 언급된 경우 두 버전 모두 출력
                if has_uv and has_python:
                    cmd = "uv --version && python --version"
                elif has_uv:
                    cmd = "uv --version"
                elif has_python:
                    cmd = "python --version"
                elif "node" in task_lower or "노드" in task_lower:
                    cmd = "node --version"
                elif "npm" in task_lower:
                    cmd = "npm --version"
                elif "git" in task_lower or "깃" in task_lower:
                    cmd = "git --version"
                else:
                    # 기본: uv와 python 버전 모두 확인
                    cmd = "uv --version && python --version"
                tool_name = "execute_command"
                arguments = {"command": cmd}
            elif any(k in task_lower for k in ["파일", "files", "폴더", "directory", "dir", "ls", "목록"]):
                # 파일/폴더 목록 확인
                tool_name = "execute_command"
                arguments = {"command": "dir /b"}  # Windows
            # 뉴스/검색 키워드 감지
            elif any(k in task_lower for k in ["news", "latest", "뉴스", "소식"]):
                tool_name = "search_news"
                arguments = {"query": task_description, "num_results": 5}
            elif any(k in task_lower for k in ["search", "검색", "찾아"]):
                tool_name = "search_web"
                arguments = {"query": task_description, "num_results": 5}
            elif any(k in task_lower for k in ["weather", "날씨", "기온"]):
                # [Fix] 도시명 추출, 없으면 Seoul 기본값
                cities = {"서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon", 
                         "광주": "Gwangju", "대전": "Daejeon", "tokyo": "Tokyo", "london": "London"}
                location = "Seoul"  # 기본값
                for k, v in cities.items():
                    if k in task_lower:
                        location = v
                        break
                tool_name = "get_weather"
                arguments = {"location": location}
            elif any(k in task_lower for k in ["time", "시간", "몇시"]):
                tool_name = "get_current_time"
                arguments = {"timezone": "Asia/Seoul"}
            else:
                # 기본: 웹 검색
                tool_name = "search_web"
                arguments = {"query": task_description, "num_results": 5}
            
            # 2. Tool 직접 실행 (Brain 모델 우회)
            if self.orchestrator.tool_executor is None:
                from tools.executor import ToolExecutor
                self.orchestrator._tool_executor = ToolExecutor()
            
            self.logger.info(f"[{self.name}] API Call: {tool_name}({arguments})")
            result = self.orchestrator.tool_executor.execute(tool_name, arguments)
            self.logger.info(f"[{self.name}] Tool task completed. Result: {str(result)[:50]}...")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] Error in ToolWorker: {e}")
            raise e


