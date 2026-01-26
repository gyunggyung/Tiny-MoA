"""
Tiny Cowork TUI Dashboard
=========================
실시간 작업 진행 상황 및 에이전트 상태를 시각화합니다.
"""

import time
from datetime import datetime
from typing import List, Optional
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

class CoworkDashboard:
    def __init__(self, goal: str = "Idle"):
        self.console = Console()
        self.layout = Layout()
        self.goal = goal
        self.tasks = [] # [{"id": "...", "desc": "...", "status": "...", "agent": "..."}]
        self.logs = []
        self.start_time = time.time()
        
        self._setup_layout()
        
    def _setup_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="task_list", ratio=2),
            Layout(name="agent_logs", ratio=3),
        )
        
    def update_tasks(self, tasks: List[dict]):
        self.tasks = tasks
        
    def add_log(self, message: str, agent: str = "System"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] [{agent}] {message}")
        if len(self.logs) > 100: # Increased log capacity further
            self.logs.pop(0)
            
    def _make_header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right", ratio=1)
        grid.add_row(
            Text.from_markup(f"[bold magenta]🤝 Tiny Cowork v2.0[/bold magenta] | Goal: [cyan]{self.goal}[/cyan]"),
            datetime.now().ctime(),
        )
        return Panel(grid, style="white on blue")

    def _make_task_list(self) -> Panel:
        table = Table(title="Task Board", expand=True)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Description", ratio=1)
        table.add_column("Agent", style="yellow")
        table.add_column("Status", style="bold")
        
        for t in self.tasks:
            status = t.get("status", "Pending")
            style = "white"
            if status == "Running": style = "cyan blink"
            elif status == "Completed": style = "green"
            elif status == "Failed": style = "red"
            
            table.add_row(
                t.get("id", "N/A"),
                t.get("desc", "N/A"),
                t.get("agent", "brain"),
                Text(status, style=style),
            )
        return Panel(table, border_style="magenta")

    def _make_logs(self) -> Panel:
        log_text = Text()
        for log in self.logs:
            if "System" in log: style = "dim"
            elif "Worker" in log: style = "green"
            elif "Planner" in log: style = "yellow"
            elif "Tool" in log: style = "cyan"
            elif "Source" in log: style = "bold white" # Articles
            elif "Error" in log: style = "bold red"
            elif "URL:" in log: style = "blue underline"
            else: style = "white"
            log_text.append(log + "\n", style=style)
            
        return Panel(log_text, title="Agent Activity Log", border_style="cyan")

    def _make_footer(self) -> Panel:
        elapsed = int(time.time() - self.start_time)
        return Panel(
            Text.from_markup(f"Elapsed: {elapsed}s | System: [green]Healthy[/green] | Mode: [bold]Autonomous[/bold]"),
            border_style="dim"
        )

    def generate_layout(self) -> Layout:
        self.layout["header"].update(self._make_header())
        self.layout["task_list"].update(self._make_task_list())
        self.layout["agent_logs"].update(self._make_logs())
        self.layout["footer"].update(self._make_footer())
        return self.layout

def demo_dashboard():
    dash = CoworkDashboard("Summarizing Project Reports")
    dash.update_tasks([
        {"id": "t1", "desc": "Read docs/report1.pdf", "status": "Completed", "agent": "rag"},
        {"id": "t2", "desc": "Extract code from src/", "status": "Running", "agent": "tool"},
        {"id": "t3", "desc": "Generate Summary", "status": "Pending", "agent": "writer"},
    ])
    
    with Live(dash.generate_layout(), refresh_per_second=4, screen=True) as live:
        for i in range(10):
            dash.add_log(f"Processing chunk {i}...", "Worker")
            time.sleep(0.5)
            live.update(dash.generate_layout())

if __name__ == "__main__":
    demo_dashboard()
