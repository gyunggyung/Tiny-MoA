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

console = Console()


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
    
    def _handle_tool_call(self, user_input: str, tool_hint: str = "", verbose: bool = True) -> str:
        """
        Tool 호출 처리
        
        1. Falcon-90M으로 JSON 생성 (또는 키워드 기반 폴백)
        2. Tool 실행
        3. Brain으로 결과 포맷팅
        """
        if not self.enable_tools or self.tool_executor is None:
            return self.brain.direct_respond(
                user_input,
                system_prompt="The user is asking about real-time information but tools are not available. Apologize and explain."
            )
        
        # 1. Tool 호출 JSON 생성
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
        
        # 2. Tool 실행
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        
        if verbose:
            console.print(f"[dim]🔨 Tool 실행: {tool_name}({arguments})[/dim]")
        
        result = self.tool_executor.execute(tool_name, arguments)
        
        if verbose:
            console.print(Panel(
                JSON.from_data(result),
                title=f"[bold cyan]🔧 {tool_name} 결과[/bold cyan]",
                border_style="cyan",
            ))
        
        # 3. Brain으로 결과 포맷팅
        if result.get("success", False):
            tool_result = result.get("result", {})
            format_prompt = f"""User asked: "{user_input}"

Tool "{tool_name}" returned this result:
{tool_result}

Please provide a natural, helpful response to the user in their language (Korean if they asked in Korean).
Be concise and format the information nicely."""
            
            return self.brain.direct_respond(
                format_prompt,
                system_prompt="You are a helpful assistant presenting tool results to users."
            )
        else:
            # Tool 실패
            error = result.get("error", "Unknown error")
            return f"죄송합니다. 정보를 가져오는 데 실패했습니다: {error}"
    
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
        
        if any(kw in user_lower for kw in search_keywords):
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
            final_response = self._handle_tool_call(user_input, tool_hint, verbose)
            
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

