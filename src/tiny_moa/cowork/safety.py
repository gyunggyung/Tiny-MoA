"""
Safety Guard
============
Tiny Cowork 시스템의 안전 장치.
파일 삭제, 포맷 등 파괴적인 동작을 감지하고 차단하거나 사용자 확인을 요청합니다.
"""

import os
from pathlib import Path
from typing import Tuple

class SafetyGuard:
    """파괴적 동작 방지 및 샌드박스 검증"""
    
    # 위험한 키워드 (파일명이나 내용이 아니라, 'Action' 이름 기준)
    DANGEROUS_ACTIONS = [
        "delete", "remove", "rm", "rmdir", "unlink",
        "format", "drop", "truncate", "overwrite"
    ]
    
    def __init__(self, sandbox_root: Path):
        self.sandbox_root = sandbox_root.resolve()
    
    def validate_path(self, path: str) -> Tuple[bool, str]:
        """
        경로가 샌드박스 내부에 있는지 확인 (Path Traversal 방지)
        """
        try:
            target = (self.sandbox_root / path).resolve()
            # 샌드박스 루트로 시작하는지 확인
            if str(target).startswith(str(self.sandbox_root)):
                return True, str(target)
            else:
                return False, f"Access Denied: Path '{path}' is outside sandbox."
        except Exception as e:
            return False, str(e)

    def check_action(self, action: str, target_path: str = "") -> Tuple[bool, str]:
        """
        위험한 동작 감지
        Returns:
            (is_safe, message)
        """
        action_lower = action.lower()
        
        # 1. 명시적 위험 키워드 확인
        for danger in self.DANGEROUS_ACTIONS:
            if danger in action_lower:
                return False, f"⚠️ Dangerous Action Detected: '{danger}'. Confirmation required."
        
        # 2. 경로 검증
        if target_path:
            is_valid_path, msg = self.validate_path(target_path)
            if not is_valid_path:
                return False, msg
                
        return True, "Safe"
