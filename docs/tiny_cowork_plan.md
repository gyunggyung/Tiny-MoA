# 🤝 Tiny Cowork 구현 계획

> **목표:** Claude Cowork에서 영감을 받은 경량 협업 에이전트 시스템 구축

---

## 📋 Claude Cowork 분석

### Cowork란?
> "Claude Code for the rest of your work" - Anthropic (2026.01.12)

Claude Cowork는 개발자가 아닌 일반 사용자도 Claude Code처럼 에이전트 방식으로 작업할 수 있게 해주는 Anthropic의 새로운 기능입니다.

### 핵심 특징
| 기능 | 설명 |
|------|------|
| **폴더 접근** | 사용자가 선택한 폴더 읽기/편집/생성 |
| **자율 실행** | 작업 계획 수립 후 자동 완료, 중간 피드백 가능 |
| **병렬 큐** | 여러 작업을 동시에 처리 |
| **스킬 확장** | 문서, 프레젠테이션 생성 등 특화 기능 |
| **브라우저 연동** | Chrome 확장과 연계 가능 |

### 사용자 경험 (Reddit 피드백)
```
"Cowork가 Claude Code를 지휘하는 모습은... 미래가 두려울 정도로 효율적"
"6개월 동안 못 만든 앱을 Cowork가 1시간 만에 완성"
"마치 동료에게 메시지를 남기는 느낌"
```

---

## 🎯 Tiny Cowork 설계

### 핵심 컨셉
```
┌─────────────────────────────────────────────────────────────┐
│                    🎯 Tiny Cowork                           │
│     "작은 모델들이 협력하는 로컬 AI 동료"                    │
└─────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ 폴더    │       │ 병렬    │       │ 에이전트│
    │ 샌드박스│       │ 태스크  │       │ 협업    │
    └─────────┘       └─────────┘       └─────────┘
```

### Phase 1: 폴더 기반 컨텍스트

```python
class WorkspaceContext:
    """사용자가 지정한 폴더의 파일 접근 관리"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.allowed_extensions = ['.txt', '.md', '.json', '.csv', '.py']
    
    def read_file(self, filename: str) -> str:
        """파일 읽기 (샌드박스 내부만)"""
        pass
    
    def write_file(self, filename: str, content: str) -> bool:
        """파일 쓰기 (확인 후)"""
        pass
    
    def list_files(self) -> list[str]:
        """폴더 내 파일 목록"""
        pass
    
    def create_file(self, filename: str, content: str) -> bool:
        """새 파일 생성"""
        pass
```

### Phase 2: 태스크 큐 시스템

```python
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Thread

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class CoworkTask:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    agent: str = "brain"  # brain, reasoner, tool, culture, rag

class TaskQueue:
    """병렬 태스크 처리 큐"""
    
    def __init__(self, max_workers: int = 3):
        self.queue = Queue()
        self.workers = []
        self.max_workers = max_workers
    
    def add_task(self, task: CoworkTask):
        """태스크 추가 (사용자가 여러 개 계속 추가 가능)"""
        self.queue.put(task)
    
    def process_tasks(self):
        """병렬로 태스크 처리"""
        pass
```

### Phase 3: 에이전트 오케스트레이션

```
사용자: "이 폴더의 PDF 정리하고, 메모 요약해서 리포트 만들어줘"

              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   📋 Planner Agent                          │
│  태스크 분해:                                                │
│  1. PDF 파일 목록 확인 → RAG Agent                          │
│  2. 각 PDF 내용 추출 → Docling Agent                        │
│  3. 메모 파일 읽기 → Workspace Context                       │
│  4. 요약 생성 → Brain Agent                                  │
│  5. 리포트 작성 → Brain Agent                                │
└─────────────────────────────────────────────────────────────┘
              │
    ┌─────────┴─────────┬─────────────────┐
    ▼                   ▼                 ▼
┌─────────┐       ┌─────────┐       ┌─────────┐
│ Task 1  │       │ Task 2  │       │ Task 3  │
│ (RAG)   │       │(Docling)│       │(Context)│
└─────────┘       └─────────┘       └─────────┘
    │                   │                 │
    └───────────────────┴─────────────────┘
                        │
                        ▼
              ┌───────────────┐
              │  Brain Agent  │
              │  (요약+리포트) │
              └───────────────┘
```

### Phase 4: 안전 장치

```python
class SafetyGuard:
    """파괴적 동작 방지"""
    
    DANGEROUS_ACTIONS = [
        "delete", "remove", "rm", "rmdir",
        "format", "drop", "truncate"
    ]
    
    def check_action(self, action: str) -> tuple[bool, str]:
        """위험한 동작 감지 및 확인 요청"""
        for danger in self.DANGEROUS_ACTIONS:
            if danger in action.lower():
                return False, f"⚠️ '{danger}' 동작 감지. 계속하시겠습니까?"
        return True, ""
    
    def require_confirmation(self, action: str) -> bool:
        """사용자 확인 요청"""
        print(f"🔒 확인 필요: {action}")
        response = input("계속? (y/n): ")
        return response.lower() == 'y'
```

---

## 📊 Claude Cowork vs Tiny Cowork

| 기능 | Claude Cowork | Tiny Cowork |
|------|---------------|-------------|
| **모델** | Claude (클라우드) | LFM2.5 + Falcon (로컬) |
| **비용** | Max 구독 ($100+/월) | 무료 (로컬) |
| **프라이버시** | 클라우드 전송 | 완전 로컬 |
| **폴더 접근** | ✅ | ✅ |
| **병렬 태스크** | ✅ | ✅ |
| **브라우저 연동** | Chrome 확장 | 미정 |
| **스킬 시스템** | ✅ | 제한적 |
| **품질** | 높음 | 중간 (작은 모델) |

---

## 📁 파일 구조

```
src/tiny_moa/
├── cowork/                    # [NEW] Tiny Cowork 모듈
│   ├── __init__.py
│   ├── workspace.py           # 폴더 컨텍스트 관리
│   ├── task_queue.py          # 태스크 큐 시스템
│   ├── planner.py             # 태스크 분해 에이전트
│   ├── orchestrator.py        # 에이전트 오케스트레이션
│   ├── safety.py              # 안전 장치
│   └── ui.py                  # 간단한 CLI/TUI
├── agents/                    # 기존 에이전트들
│   ├── brain.py
│   ├── reasoner.py
│   ├── tool_caller.py
│   ├── culture_agent.py
│   └── rag_agent.py
└── main.py
```

---

## 🔄 실행 흐름 예시

### 입력: "다운로드 폴더 정리해줘"

```
1. 📂 Workspace 스캔
   └─ downloads/ 폴더 파일 목록 확인

2. 📋 Planner 분석
   └─ 파일 타입별 분류 계획 수립
   └─ 하위 태스크 생성: [이미지 분류, 문서 분류, 기타]

3. 🔄 병렬 실행
   ├─ Task1: 이미지 → images/ 이동
   ├─ Task2: PDF/DOCX → documents/ 이동
   └─ Task3: ZIP → archives/ 이동

4. 🔒 안전 확인
   └─ "23개 파일을 이동합니다. 계속?" → 사용자 확인

5. ✅ 완료 리포트
   └─ "정리 완료: 이미지 12개, 문서 8개, 압축파일 3개"
```

---

## ⏱️ 예상 일정

| 단계 | 작업 | 예상 시간 |
|------|------|-----------|
| 1 | Workspace Context 구현 | 2시간 |
| 2 | Task Queue 시스템 | 2시간 |
| 3 | Planner Agent | 2시간 |
| 4 | 에이전트 오케스트레이션 | 3시간 |
| 5 | Safety Guard | 1시간 |
| 6 | CLI/TUI 인터페이스 (Rich/Textual) | 3시간 |
| 7 | 테스트 및 디버깅 | 2시간 |
| **총계** | | **~15시간** |

---

## 🖥️ TUI 구현 계획 (Text User Interface)

**사용 라이브러리**: `rich` (출력), `prompt_toolkit` 또는 `textual` (인터랙션)

### 화면 구성 안
```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Tiny Cowork                                      [v0.1] │
├────────────────────────────┬────────────────────────────────┤
│ 📂 Workspace: /src         │ 🤖 Agent Status                │
│ [ ] main.py                │                                │
│ [x] utils.py               │ [Busy] Brain: 요약 중...       │
│                            │ [Idle] Planner                 │
├────────────────────────────┤ [Idle] Tool                    │
│ 📝 Task Queue              │                                │
│ 1. [▶] 코드 리뷰           │                                │
│ 2. [ ] 문서화              │                                │
├────────────────────────────┴────────────────────────────────┤
│ 💬 Log / Chat                                               │
│ > main.py 파일 분석을 시작합니다...                          │
│ > 함수 3개 발견.                                            │
│                                                             │
│ [Input] ___________________________________________________ │
└─────────────────────────────────────────────────────────────┘
```

1.  **Dashboard**: 현재 작업 상태, 에이전트 상태, 파일 목록 한눈에 보기
2.  **Live Updates**: 비동기 상태 업데이트 (spinner, progress bar)
3.  **Command Input**: 자연어 명령 또는 단축키 지원

---

## ⚠️ 주의사항 (Claude Cowork 참고)

1. **Prompt Injection 위험**: 외부 문서에 악성 지시가 포함될 수 있음
2. **파괴적 동작**: 파일 삭제 등 위험 동작 전 반드시 확인
3. **샌드박스 제한**: 지정된 폴더 외부 접근 차단
4. **컨텍스트 오버플로우**: 큰 폴더 처리 시 메모리 관리 필요

---

## 📚 참고 자료

- [Claude Cowork 공식 발표](https://claude.com/blog/cowork-research-preview) (2026.01.12)
- [Reddit: Claude Cowork 사용기](https://reddit.com/r/ClaudeCode)
- [Claude Code 아키텍처](https://docs.anthropic.com/claude-code)

---

## 🔮 향후 확장

### v2.0 계획
- **웹 UI**: Streamlit/Gradio 기반 웹 인터페이스
- **스킬 시스템**: 사용자 정의 스킬 추가
- **히스토리**: 작업 이력 저장 및 재실행
- **멀티 폴더**: 여러 폴더 동시 작업
- **클라우드 동기화**: 설정/스킬 동기화 (옵션)
