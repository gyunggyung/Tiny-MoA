"""
Workspace Context
=================
에이전트가 작업하는 파일 시스템 공간(Sandbox) 관리.
"""

import os
from pathlib import Path
from typing import List, Optional
from src.tiny_moa.cowork.safety import SafetyGuard

class WorkspaceContext:
    """사용자가 지정한 폴더의 파일 접근 관리 (Sandbox)"""
    
    def __init__(self, root_path: str = "./workspace"):
        self.root_path = Path(root_path).resolve()
        self.root_path.mkdir(exist_ok=True, parents=True)
        self.guard = SafetyGuard(self.root_path)
        
    def list_files(self, recursive: bool = False) -> List[str]:
        """
        폴더 내 파일 목록 반환 (상대 경로)
        """
        files = []
        if recursive:
            for p in self.root_path.rglob("*"):
                if p.is_file():
                    files.append(str(p.relative_to(self.root_path)))
        else:
            for p in self.root_path.glob("*"):
                if p.is_file():
                    files.append(p.name)
        return files
    
    def read_file(self, filename: str) -> str:
        """파일 읽기"""
        is_safe, msg = self.guard.validate_path(filename)
        if not is_safe:
            return f"Error: {msg}"
            
        target = self.root_path / filename
        if not target.exists():
            return f"Error: File not found '{filename}'"
            
        try:
            # 텍스트 파일만 읽기 시도 (바이너리 제외)
            return target.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, filename: str, content: str) -> str:
        """파일 쓰기 (Overwrite)"""
        is_safe, msg = self.guard.validate_path(filename)
        if not is_safe:
            return f"Error: {msg}"
            
        target = self.root_path / filename
        
        # 위험 동작 체크 (여기서는 단순 쓰기이지만, 덮어쓰기 경고 등 추가 가능)
        # 현재는 항상 허용하되, 로그 남김
        
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return f"Successfully wrote to '{filename}'"
        except Exception as e:
            return f"Error writing file: {e}"

    def get_context_description(self) -> str:
        """현재 작업 공간 상태 요약 (LLM 입력용)"""
        files = self.list_files(recursive=True)
        file_list_str = "\n".join([f"- {f}" for f in files[:20]]) # 최대 20개만 표시
        if len(files) > 20:
            file_list_str += f"\n... (and {len(files)-20} more)"
            
        return f"""
Current Workspace: {self.root_path}
Files:
{file_list_str}
"""
