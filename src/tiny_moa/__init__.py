"""
Tiny MoA (Mixture of Agents) PoC
================================
GPU Poor를 위한 AI 군단 - 1.2B Brain + 600M Specialist

사용법:
    from tiny_moa import TinyMoA
    
    moa = TinyMoA()
    response = moa.chat("피보나치 함수 작성해줘")
"""

from tiny_moa.orchestrator import TinyMoA

__version__ = "0.1.0"
__all__ = ["TinyMoA"]
