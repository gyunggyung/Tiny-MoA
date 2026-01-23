"""
Tiny MoA 오케스트레이터
======================
Brain과 Specialist를 조율하여 사용자 요청 처리
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from tiny_moa.brain import Brain
from tiny_moa.reasoner import Reasoner

console = Console()


class TinyMoA:
    """Tiny MoA (Mixture of Agents) 오케스트레이터"""
    
    def __init__(
        self,
        brain_path: Optional[str] = None,
        reasoner_path: Optional[str] = None,
        n_ctx: int = 4096,
        use_thinking: bool = False,
        lazy_load: bool = True,
    ):
        """
        Args:
            brain_path: Brain 모델 경로
            reasoner_path: Reasoner 모델 경로
            n_ctx: 컨텍스트 길이
            use_thinking: LFM Thinking 모델 사용 여부 (실험 중)
            lazy_load: Reasoner를 첫 사용 시 로드할지 여부
        """
        self.brain_path = brain_path
        self.reasoner_path = reasoner_path
        self.n_ctx = n_ctx
        self.use_thinking = use_thinking
        self.lazy_load = lazy_load
        
        self._brain: Optional[Brain] = None
        self._reasoner: Optional[Reasoner] = None
        
        console.print("[bold blue]🤖 Tiny MoA 초기화 중...[/bold blue]")
        
        # Brain은 항상 로드 (라우터 역할)
        self._load_brain()
        
        # Reasoner는 lazy_load 설정에 따라
        if not lazy_load:
            self._load_reasoner()
        
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
        
        # 1. Brain이 라우팅 결정
        route_result = self.brain.route(user_input)
        route = route_result.get("route", "DIRECT")
        specialist_prompt = route_result.get("specialist_prompt", "")
        
        if verbose:
            console.print(f"[dim]🧠 라우팅: {route}[/dim]")
        
        # 2. 라우팅에 따른 처리
        if route == "REASONER" and specialist_prompt:
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
            final_response = self.brain.direct_respond(user_input)
        
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
        "종료하려면 'quit' 또는 'exit' 입력",
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
