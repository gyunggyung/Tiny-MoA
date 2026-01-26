"""
Tiny MoA 오케스트레이터
======================
Brain과 Specialist를 조율하여 사용자 요청 처리
Tool Calling 지원 추가
"""

import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.json import JSON
import re

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tiny_moa.brain import Brain
from tiny_moa.reasoner import Reasoner

# 번역 모듈 import
try:
    from translation.pipeline import TranslationPipeline
    from translation.detector import detect_language
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

console = Console(force_terminal=True, color_system="auto")


class TinyMoA:
    """Tiny MoA (Mixture of Agents) 오케스트레이터"""
    
    def __init__(
        self,
        brain_path: Optional[str] = None,
        reasoner_path: Optional[str] = None,
        tool_caller_path: Optional[str] = None,
        n_ctx: int = 4096,
        use_thinking: bool = False,
        lazy_load: bool = True,
        enable_tools: bool = True,
        enable_translation: bool = True,
    ):
        """
        Args:
            brain_path: Brain 모델 경로
            reasoner_path: Reasoner 모델 경로
            tool_caller_path: Tool Caller (Falcon-90M) 경로
            n_ctx: 컨텍스트 길이
            use_thinking: LFM Thinking 모델 사용 여부 (실험 중)
            lazy_load: Reasoner/ToolCaller를 첫 사용 시 로드할지 여부
            enable_tools: Tool Calling 기능 활성화 여부
        """
        self.brain_path = brain_path
        self.reasoner_path = reasoner_path
        self.tool_caller_path = tool_caller_path
        self.n_ctx = n_ctx
        self.use_thinking = use_thinking
        self.lazy_load = lazy_load
        self.enable_tools = enable_tools
        self.enable_translation = enable_translation and TRANSLATION_AVAILABLE
        
        # 번역 파이프라인 초기화
        self._translation_pipeline = None
        if self.enable_translation:
            try:
                self._translation_pipeline = TranslationPipeline(use_simple_translator=True)
                console.print("[dim]🌐 Translation Pipeline 활성화[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠️ 번역 비활성화: {e}[/yellow]")
                self.enable_translation = False
        
        self._brain: Optional[Brain] = None
        self._reasoner: Optional[Reasoner] = None
        self._tool_caller = None
        self._tool_executor = None
        
        console.print("[bold blue]🤖 Tiny MoA 초기화 중...[/bold blue]")
        
        # Brain은 항상 로드 (라우터 역할)
        self._load_brain()
        
        # Reasoner/ToolCaller는 lazy_load 설정에 따라
        if not lazy_load:
            self._load_reasoner()
            if enable_tools:
                self._load_tool_caller()
        
        console.print("[bold green]✅ Tiny MoA 준비 완료![/bold green]")
    
    def _load_brain(self):
        """Brain 모델 로드"""
        if self._brain is None:
            console.print("[dim]Loading Brain (LFM2.5-1.2B)...[/dim]")
            self._brain = Brain(
                model_path=self.brain_path,
                n_ctx=self.n_ctx,
                use_thinking=self.use_thinking,
            )
    
    def _load_reasoner(self):
        """Reasoner 모델 로드 (Lazy)"""
        if self._reasoner is None:
            console.print("[dim]Loading Reasoner (Falcon-R-0.6B)...[/dim]")
            self._reasoner = Reasoner(
                model_path=self.reasoner_path,
                n_ctx=self.n_ctx,
            )
    
    def _load_tool_caller(self):
        """Tool Caller 로드 (Lazy)"""
        if self._tool_caller is None and self.enable_tools:
            try:
                from tools.caller import ToolCaller
                from tools.executor import ToolExecutor
                
                console.print("[dim]Loading Tool Caller (Falcon-90M)...[/dim]")
                self._tool_caller = ToolCaller(
                    falcon_path=self.tool_caller_path,
                    brain_model=self._brain,  # Brain으로 JSON 보정
                )
                self._tool_executor = ToolExecutor()
                console.print("[dim]✅ Tool Caller 준비 완료[/dim]")
            except ImportError as e:
                console.print(f"[yellow]⚠️ Tool Calling 비활성화: {e}[/yellow]")
                self.enable_tools = False
    
    @property
    def brain(self) -> Brain:
        if self._brain is None:
            self._load_brain()
        return self._brain
    
    @property
    def reasoner(self) -> Reasoner:
        if self._reasoner is None:
            self._load_reasoner()
        return self._reasoner
    
    @property
    def tool_caller(self):
        if self._tool_caller is None:
            self._load_tool_caller()
        return self._tool_caller
    
    @property
    def tool_executor(self):
        if self._tool_executor is None:
            self._load_tool_caller()
        return self._tool_executor
    
    def _handle_tool_call(self, user_input: str, tool_hint: str = "", arg_hint: str = "", verbose: bool = True) -> str:
        """
        Tool 호출 처리
        
        1. Brain이 제공한 arg_hint가 있으면 우선 사용
        2. 아니면 Falcon-90M으로 JSON 생성 (또는 키워드 기반 폴백)
        3. Tool 실행
        4. Brain으로 결과 포맷팅
        """
        if not self.enable_tools or self.tool_executor is None:
            return self.brain.direct_respond(
                user_input,
                system_prompt="The user is asking about real-time information but tools are not available. Apologize and explain."
            )
        
        tool_call = {}
        
        # 1. Brain이 제공한 최적화 인자 사용 (우선순위 1)
        if arg_hint and tool_hint:
            if verbose:
                console.print(f"[dim]🧠 Brain 최적화 인자 사용: {tool_hint}({arg_hint})[/dim]")
            
            arguments = {}
            if tool_hint in ["search_web", "search_news", "search_wikipedia"]:
                arguments = {"query": arg_hint}
            elif tool_hint == "execute_command":
                # 방어 로직: 명령어가 자연어 문장으로 보이면 무시하고 키워드 폴백 사용
                # LFM 1.2B가 가끔 "Check if..." 같은 지시문을 생성함
                is_valid_cmd = True
                bad_starters = ["Check", "Verify", "Confirm", "Please", "Ensure", "See", "Test", "Determine"]
                
                # 1. 자연어 시작 패턴 체크
                if any(arg_hint.strip().startswith(s) for s in bad_starters) and len(arg_hint.split()) > 2:
                    is_valid_cmd = False
                
                # 2. 한글 포함 여부 체크 (명령어에 한글이 있으면 자연어 설명일 확률 높음)
                if re.search(r'[가-힣]', arg_hint):
                    is_valid_cmd = False
                
                if is_valid_cmd:
                    arguments = {"command": arg_hint}
                else:
                    if verbose:
                         console.print(f"[yellow]⚠️ Brain 생성 명령어('{arg_hint}')가 자연어 설명으로 감지되어 무시합니다. 키워드 추론을 사용합니다.[/yellow]")
                    # arguments를 비워두면 아래쪽 tool_call 생성 조건(if arguments:)을 만족하지 못해
                    # 자연스럽게 2. Falcon/키워드 폴백 로직으로 넘어감
                    arguments = {}
            elif tool_hint == "get_weather":
                arguments = {"location": arg_hint}
            elif tool_hint == "get_current_time":
                arguments = {"timezone": arg_hint}
            elif tool_hint == "calculate":
                arguments = {"expression": arg_hint}
            elif tool_hint == "read_url":
                arguments = {"url": arg_hint}
            
            if arguments:
                tool_call = {"name": tool_hint, "arguments": arguments}
        
        # 2. Tool Call이 아직 없으면 Falcon/키워드 사용
        if not tool_call:
            if self.tool_caller and self.tool_caller._falcon:
                # Falcon-90M 사용
                if verbose:
                    console.print("[dim]🔧 Tool Caller (Falcon-90M) 호출 중...[/dim]")
                tool_call = self.tool_caller.generate_tool_call(user_input)
            else:
                # 키워드 기반 폴백 (모델 없이)
                if verbose:
                    console.print("[dim]🔧 키워드 기반 Tool 추론 중...[/dim]")
                tool_call = self._infer_tool_from_keywords(user_input, tool_hint)
        
        if "error" in tool_call:
            if verbose:
                console.print(f"[yellow]⚠️ Tool 파싱 실패: {tool_call['error']}[/yellow]")
            return self.brain.direct_respond(user_input)
        
        # 2. Tool 실행 (Retry Logic)
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        
        if verbose:
            console.print(f"[dim]🔨 Tool 실행: {tool_name}({arguments})[/dim]")
        
        result = self.tool_executor.execute(tool_name, arguments)
        
        if verbose:
            console.print(Panel(
                JSON.from_data(result),
                title=f"[bold cyan]🔧 {tool_name} { '성공' if result.get('success') else '실패' }[/bold cyan]",
                border_style="cyan" if result.get("success") else "red",
            ))
        
        # [Semantic Error Detection] Soft Error 감지
        # 툴이 성공(True)했다고 보고해도, 내용에 에러 키워드가 있으면 실패로 간주
        if result.get("success", False):
            raw_result = str(result.get("result", "")).lower()
            error_keywords = ["timeout", "timed out", "rate limit", "api error", "access denied", "404 not found", "500 internal server error", "traceback"]
            
            # 단, "error"라는 단어는 일반 문장에도 들어갈 수 있으므로 주의 (여기서는 보수적으로 제외하거나 문맥 파악 필요)
            # 확실한 시스템 에러 키워드만 우선 적용
            
            for keyword in error_keywords:
                if keyword in raw_result:
                    if verbose:
                        console.print(f"[yellow]⚠️ Semantic Error 감지: '{keyword}' - 재시도 트리거[/yellow]")
                    result["success"] = False
                    result["error"] = f"Tool returned success but contained error keyword: {keyword}"
                    break
        
        # 3. Brain으로 결과 포맷팅 or 재시도
        if result.get("success", False):
            tool_result = result.get("result", {})
            # Brain의 integrate_response를 사용하여 환각 방지 및 포맷팅 적용
            return self.brain.integrate_response(user_input, str(tool_result))
        else:
            # Tool 실패 -> 재시도 (Retry)
            error = result.get("error", "Unknown error")
            
            # 모든 Tool 실패 시 1회 재시도 (Brain에게 수정 요청)
            if "retry" not in arguments: # 무한 루프 방지
                if verbose:
                    console.print(f"[bold red]⚠️ 실행 실패: {error}. Brain에게 수정을 요청합니다...[/bold red]")
                
                # Brain에게 수정을 요청하는 프롬프트
                retry_prompt = f"""The tool '{tool_name}' failed with arguments '{arguments}'
Error: "{error}".
The user wants to: "{user_input}".
Please provide CORRECTED arguments for the tool '{tool_name}' to fix this error.
Return ONLY the JSON arguments (e.g. {{"location": "Seoul"}} or {{"command": "python --version"}}). Do NOT explain."""

                corrected_args_str = self.brain.direct_respond(
                    retry_prompt, 
                    system_prompt="You are a tool expert. Provide only the corrected JSON arguments."
                ).strip()
                
                # 마크다운/JSON 파싱 시도
                corrected_args_str = corrected_args_str.replace("```json", "").replace("```", "").strip()
                
                try:
                    import json
                    # 단순 문자열인 경우(예: command string) 처리
                    if not corrected_args_str.startswith("{"):
                         # execute_command라면 문자열을 command로 간주
                         if tool_name == "execute_command":
                             retry_args = {"command": corrected_args_str}
                         else:
                             # 다른 툴은 location 등 키를 알기 어려우므로 JSON 파싱 재시도하거나 포기
                             # 여기서는 간단히 location이나 query로 가정하는 휴리스틱 추가 가능하나,
                             # Brain이 JSON을 주도록 프롬프트했으므로 일단 JSON 로드 시도
                             pass
                    
                    if corrected_args_str.startswith("{"):
                        retry_args = json.loads(corrected_args_str)
                        retry_args["retry"] = True # 재귀 방지 플래그
                        
                        if verbose:
                            console.print(f"[dim]🧠 Brain 수정 제안: {retry_args}[/dim]")

                        retry_result = self.tool_executor.execute(tool_name, retry_args)
                        
                        if verbose:
                            console.print(Panel(
                                JSON.from_data(retry_result),
                                title=f"[bold cyan]🔧 재시도 결과[/bold cyan]",
                                border_style="cyan" if retry_result.get("success") else "red",
                            ))
                            
                        if retry_result.get("success"):
                            # 성공 시 포맷팅 후 반환
                            tool_result = retry_result.get("result", {})
                            # Brain의 integrate_response를 사용하여 환각 방지 및 포맷팅 적용
                            return self.brain.integrate_response(user_input, str(tool_result))
                        else:
                            error = retry_result.get("error", error)
                except Exception as e:
                    if verbose:
                        console.print(f"[dim]⚠️ 재시도 파싱 실패: {e}[/dim]")

            return f"죄송합니다. 명령 실행에 실패했습니다.\n오류: {error}"
    
    def _infer_tool_from_keywords(self, user_input: str, tool_hint: str = "") -> dict:
        """키워드 기반 Tool 호출 추론 (모델 없이)"""
        user_lower = user_input.lower()
        
        # tool_hint가 있으면 우선 사용
        if tool_hint == "get_weather":
            # 도시명 추출 시도
            cities = ["서울", "seoul", "도쿄", "tokyo", "뉴욕", "new york", "런던", "london", 
                      "부산", "busan", "인천", "대구", "대전", "광주"]
            location = "Seoul"  # 기본값
            for city in cities:
                if city in user_lower:
                    location = city.title()
                    break
            return {"name": "get_weather", "arguments": {"location": location}}
        
        elif tool_hint == "search_web":
            # 검색어 추출 (간단한 휴리스틱)
            query = user_input
            for prefix in ["검색해줘", "찾아봐", "알려줘", "뭐야", "search for", "search"]:
                if prefix in user_lower:
                    query = user_input.replace(prefix, "").strip()
                    break
            return {"name": "search_web", "arguments": {"query": query}}
        
        elif tool_hint == "get_current_time":
            # 타임존 추출
            timezone = "Asia/Seoul"  # 기본값
            if "뉴욕" in user_lower or "new york" in user_lower:
                timezone = "America/New_York"
            elif "도쿄" in user_lower or "tokyo" in user_lower:
                timezone = "Asia/Tokyo"
            elif "런던" in user_lower or "london" in user_lower:
                timezone = "Europe/London"
            return {"name": "get_current_time", "arguments": {"timezone": timezone}}
        
        elif tool_hint == "calculate":
            # 수식 추출
            import re
            match = re.search(r'[\d\s+\-*/().]+', user_input)
            expression = match.group().strip() if match else "0"
            return {"name": "calculate", "arguments": {"expression": expression}}
        
        # tool_hint 없을 때 키워드 기반 폴백 (영어 키워드 포함)
        weather_keywords = ["weather", "날씨", "기온", "온도", "temperature"]
        search_keywords = ["search", "find", "검색", "찾아", "알려줘"]
        time_keywords = ["time", "시간", "몇시", "what time", "current time"]
        
        if any(kw in user_lower for kw in weather_keywords):
            # 도시명 추출
            cities = ["seoul", "서울", "tokyo", "도쿄", "new york", "뉴욕", "london", "런던",
                      "busan", "부산", "incheon", "인천", "osaka", "오사카"]
            location = "Seoul"
            for city in cities:
                if city in user_lower:
                    location = city.title().replace("서울", "Seoul").replace("도쿄", "Tokyo").replace("뉴욕", "New York").replace("런던", "London").replace("부산", "Busan")
                    break
            return {"name": "get_weather", "arguments": {"location": location}}
        
        command_keywords = ["실행", "run", "check", "verify", "version", "버전", "확인", "ls", "dir", "command"]
        if any(kw in user_lower for kw in command_keywords) and ("코드" not in user_lower):
             # 간단한 명령어 추출 시도 (매우 단순화됨)
            cmd = "ver" # 기본값
            if "uv" in user_lower:
                 cmd = "uv --version"
            elif "python" in user_lower:
                 cmd = "python --version"
            elif "dir" in user_lower or "목록" in user_lower:
                 cmd = "dir"
            return {"name": "execute_command", "arguments": {"command": cmd}}

        if any(kw in user_lower for kw in search_keywords) or "uv" in user_lower:
            return {"name": "search_web", "arguments": {"query": user_input}}
        
        if any(kw in user_lower for kw in time_keywords):
            return {"name": "get_current_time", "arguments": {"timezone": "Asia/Seoul"}}
        
        return {"error": "Could not infer tool from keywords"}
    
    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        사용자 입력 처리
        
        Args:
            user_input: 사용자 메시지
            verbose: 처리 과정 출력 여부
            
        Returns:
            최종 응답
        """
        if verbose:
            console.print(f"\n[bold]📝 입력:[/bold] {user_input}")

        # 0.1. [RAG] 파일 참조 감지 (@[filename])
        # 패턴: @[filename] (공백 포함 가능)
        rag_context = ""
        rag_files = re.findall(r"@\[(.*?)\]", user_input)
        
        if rag_files:
            if verbose:
                console.print(f"[dim]📚 RAG 파일 감지: {rag_files}[/dim]")
            
            # Lazy Loading check
            if not hasattr(self, "_rag_engine") or self._rag_engine is None:
                try:
                    from src.rag.engine import RAGEngine
                    self._rag_engine = RAGEngine()
                except ImportError as e:
                     console.print(f"[red]⚠️ RAG Engine 로드 실패: {e}[/red]")
                     self._rag_engine = None

            if self._rag_engine:
                for file_ref in rag_files:
                    # 파일 경로 보정 (현재 디렉토리 기준)
                    file_path = file_ref.strip()
                    if not Path(file_path).exists():
                         # 혹시 절대 경로가 아니라면 현재 작업 디렉토리에서 찾기
                         file_path = str(Path(project_root) / file_ref.strip())
                    
                    if Path(file_path).exists():
                        # 1. Ingest (이미 처리된 경우 스킵됨 - Engine 내부 로직)
                        if verbose:
                             console.print(f"[dim]🔄 문서 처리 중: {Path(file_path).name}...[/dim]")
                        status = self._rag_engine.ingest_file(file_path)
                        if verbose:
                             console.print(f"[dim]   Result: {status}[/dim]")
                        
                        # 2. Query (질문과 관련된 내용 검색)
                        # 질문에서 파일 참조 제거 후 검색
                        clean_query = re.sub(r"@\[(.*?)\]", "", user_input).strip()
                        retrieved = self._rag_engine.query(clean_query)
                        
                        if retrieved:
                             rag_context += f"\n\n[Context from {file_ref}]\n{retrieved}\n"
                    else:
                        if verbose:
                             console.print(f"[yellow]⚠️ 파일을 찾을 수 없음: {file_ref}[/yellow]")
            
            if rag_context:
                if verbose:
                     console.print(f"[dim]📄 RAG 컨텍스트 추가됨 ({len(rag_context)} chars)[/dim]")
                # [Fix] 사용자 입력에서 @[...] 패턴 제거하여 Brain이 검색어로 오인하지 않게 함
                user_input = re.sub(r"@\[(.*?)\]", "", user_input).strip()
                
                # 사용자 입력에 컨텍스트 주입 (Brain이 읽도록)
                # 원본 질문은 유지하되, 컨텍스트를 뒤에 붙임
                user_input += f"\n\n--- Reference Material ---\n{rag_context}\n--------------------------\n(Answer strictly based on the Reference Material above if relevant.)"
        
        # 0.5. [Multi-Step] 복합 질문 분해 (Decomposition)
        # "비교", "compare", "vs" 등 키워드가 있으면 분해 시도
        complex_keywords = ["비교", "compare", "vs", "difference", "차이", "어때?"] # '어때?'는 애매하지만 일단 테스트
        is_complex = any(k in user_input for k in ["비교", "compare", "vs", "difference", "차이"])
        
        if is_complex:
            if verbose:
                console.print("[dim]🧩 복합 질문 감지: 분해 시도 중...[/dim]")
            
            sub_queries = self.brain.decompose_query(user_input)
            
            # 분해가 실제로 일어났는지 확인 (1개 이상이고, 원본과 다을 때)
            if len(sub_queries) > 1:
                if verbose:
                    console.print(f"[dim]🧩 분해 결과: {sub_queries}[/dim]")
                
                context_results = []
                for sub_q in sub_queries:
                    # 각 하위 질문 처리
                    # 재귀 호출 방지를 위해 단순 처리 로직 필요하나, 여기서는 chat() 호출하되
                    # 무한 루프 방지를 위해 is_complex 체크가 중요함.
                    # 하지만 sub_q는 단순할 것이므로 괜찮음.
                    # 다만 chat()은 번역/출력을 또 하므로, 내부 함수 _process_single_turn 같은게 필요.
                    # 여기서는 간단히: route -> handle_tool_call 복붙 로직 사용 (함수 분리 권장하지만 일단 인라인)
                    
                    # 1. Brain이 라우팅 결정 (Sub query)
                    # 번역 필요시 번역
                    sub_processed = sub_q
                    if self.enable_translation and self._translation_pipeline:
                        t_ctx = self._translation_pipeline.to_english(sub_q)
                        if t_ctx.is_translated:
                            sub_processed = t_ctx.english_text

                    route_result = self.brain.route(sub_processed)
                    route = route_result.get("route", "DIRECT")
                    
                    step_result = ""
                    if route == "TOOL":
                         tool_hint = route_result.get("tool_hint", "")
                         arg_hint = route_result.get("specialist_prompt", "")
                         # Tool 실행 및 결과 획득 (포맷팅 전의 Raw Result가 필요하지만, _handle_tool_call은 포맷팅된 텍스트 반환)
                         # 여기선 _handle_tool_call의 결과를 그대로 텍스트로 사용
                         step_result = self._handle_tool_call(sub_q, tool_hint, arg_hint, verbose=True)
                    else:
                         step_result = self.brain.direct_respond(sub_processed)
                    
                    
                    context_results.append(f"Query: {sub_q}\nResult: {step_result[:500]}") # 결과 길이 제한 (500자)
                
                # 결과 통합
                aggregated_context = "\n\n".join(context_results)
                
                # 통합 호출 전 메모리 정리 (간접적)
                if hasattr(self.brain.model, "reset"):
                    self.brain.model.reset()
                    
                final_response = self.brain.integrate_response(user_input, aggregated_context)
                
                if verbose:
                    console.print(Panel(
                        Markdown(final_response),
                        title="[bold green]💬 통합 응답[/bold green]",
                        border_style="green",
                    ))
                
                # 번역: en → original_lang (있다면)
                # 주의: decomposition 로직 시작 전에 translation_ctx를 구했어야 함.
                # 하지만 구조상 chat 함수의 메인 파이프라인(0번 단계)보다 먼저 실행됨.
                # 따라서 여기서 별도로 detect/translate 하거나, 0번 단계를 위로 올려야 함.
                # 리팩토링 최소화를 위해 여기서 간단히 처리.
                
                # (이미 chat 함수 진입 시점에는 processed_input이 없으므로, user_input을 이용)
                if self.enable_translation and self._translation_pipeline:
                     # 이미 decomposed된 쿼리는 내부적으로 번역되어 처리되었음.
                     # 최종 결과만 번역하면 됨.
                     # 단, 타겟 언어를 알기 위해 user_input 감지 필요
                     target_lang_ctx = self._translation_pipeline.to_english(user_input)
                     if target_lang_ctx.is_translated:
                          final_response = self._translation_pipeline.from_english(final_response, target_lang_ctx)
                          if verbose:
                              console.print(f"[dim]🌐 최종 번역: en → {target_lang_ctx.original_lang}[/dim]")
                              console.print(Panel(
                                    Markdown(final_response),
                                    title="[bold green]💬 번역된 응답[/bold green]",
                                    border_style="green",
                                ))

                return final_response

        # 0. 번역 파이프라인: 다국어 → 영어
        translation_ctx = None
        processed_input = user_input
        
        if self.enable_translation and self._translation_pipeline:
            translation_ctx = self._translation_pipeline.to_english(user_input)
            if translation_ctx.is_translated:
                processed_input = translation_ctx.english_text
                if verbose:
                    console.print(f"[dim]🌐 번역: {translation_ctx.original_lang} → en[/dim]")
                    console.print(f"[dim]   영어: {processed_input[:50]}...[/dim]")
        
        # 1. Brain이 라우팅 결정 (영어로 된 입력 사용)
        # [Fix] RAG 컨텍스트가 있으면 Tool Calling을 방지하고 강제로 DIRECT 응답 유도
        if rag_context:
             if verbose:
                 console.print("[dim]📄 RAG 컨텍스트 존재: 강제로 DIRECT 모드 전환[/dim]")
             route = "DIRECT"
             specialist_prompt = ""
             tool_hint = ""
        else:
             route_result = self.brain.route(processed_input)
             route = route_result.get("route", "DIRECT")
             specialist_prompt = route_result.get("specialist_prompt", "")
             tool_hint = route_result.get("tool_hint", "")
        
        if verbose:
            console.print(f"[dim]🧠 라우팅: {route}[/dim]")
        
        # 2. 라우팅에 따른 처리
        if route == "TOOL":
            # Tool Calling
            if verbose:
                console.print(f"[dim]🔧 Tool 호출: {tool_hint}[/dim]")
            # specialist_prompt를 arg_hint로 전달
            final_response = self._handle_tool_call(user_input, tool_hint, specialist_prompt, verbose)
            
        elif route == "REASONER" and specialist_prompt:
            # Reasoner 호출
            if verbose:
                console.print("[dim]🤔 Reasoner 호출 중...[/dim]")
            
            specialist_output = self.reasoner.solve(specialist_prompt)
            
            # PoC: Reasoner 출력 직접 반환 (토큰 절약)
            final_response = specialist_output
        else:
            # Brain이 직접 응답
            if verbose:
                console.print("[dim]🧠 Brain 직접 응답...[/dim]")
            final_response = self.brain.direct_respond(processed_input)
        
        # 3. 번역 파이프라인: 영어 → 원래 언어
        if translation_ctx and translation_ctx.is_translated and self._translation_pipeline:
            if verbose:
                console.print(f"[dim]🌐 번역: en → {translation_ctx.original_lang}[/dim]")
            final_response = self._translation_pipeline.from_english(final_response, translation_ctx)
        
        if verbose:
            console.print(Panel(
                Markdown(final_response),
                title="[bold green]💬 응답[/bold green]",
                border_style="green",
            ))
        
        return final_response


def interactive_mode():
    """대화형 모드"""
    console.print(Panel(
        "[bold]🤖 Tiny MoA 대화형 모드[/bold]\n"
        "🔧 Tool Calling: 날씨, 검색, 계산, 시간\n"
        "🌐 다국어 지원: 한국어, 일본어, 중국어 등\n"
        "종료: 'quit' 또는 'exit'",
        border_style="blue",
    ))
    
    moa = TinyMoA()
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[dim]👋 안녕히 가세요![/dim]")
                break
            
            if not user_input.strip():
                continue
            
            moa.chat(user_input)
            
        except KeyboardInterrupt:
            console.print("\n[dim]👋 안녕히 가세요![/dim]")
            break


if __name__ == "__main__":
    interactive_mode()

