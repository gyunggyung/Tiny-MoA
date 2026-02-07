# 💾 Tiny-MoA Memory System Plan

> **대화 기록 저장/불러오기 메모리 기능 구현 계획**  
> 참고: [Reference/openclaw](../Reference/openclaw/src/memory/)

---

## 📋 개요

### 목표
- 대화 기록을 Markdown 파일로 저장하고 불러오는 메모리 기능 구현
- 이전 대화 컨텍스트를 활용하여 연속적인 대화 가능
- GPU-Poor 환경에서도 효율적으로 동작하는 경량화된 설계

### OpenClaw 참고 사항
| 기능 | OpenClaw 구현 | Tiny-MoA 적용 |
|------|--------------|--------------|
| 세션 저장 | JSONL 형식 (`*.jsonl`) | Markdown 형식 (`*.md`) |
| 인덱싱 | SQLite + FTS5 | 단순 파일 기반 (v1) |
| 임베딩 | OpenAI/Gemini API | 로컬 Sentence Transformer (v2) |
| 검색 | 벡터 + FTS 하이브리드 | 키워드 매칭 (v1) |

---

## 🏗️ 아키텍처

```
memory/
├── sessions/                    # 대화 세션 저장
│   ├── 2026-02-07_weather.md   # 개별 세션 파일
│   ├── 2026-02-07_news.md
│   └── ...
├── MEMORY.md                    # 수동 메모/중요 정보
└── index.json                   # 세션 메타데이터 인덱스
```

### 파일 형식

#### 세션 파일 (`sessions/*.md`)
```markdown
---
session_id: abc123
created_at: 2026-02-07T10:00:00+09:00
updated_at: 2026-02-07T10:30:00+09:00
summary: "서울 날씨 조회 및 비교"
tags: [weather, seoul, tokyo]
---

## User
서울과 도쿄 날씨 비교해줘

## Assistant
서울은 -2°C, 도쿄는 8°C입니다...

## User
내일은 어때?

## Assistant
내일 서울은...
```

#### 인덱스 파일 (`index.json`)
```json
{
  "sessions": [
    {
      "id": "abc123",
      "file": "2026-02-07_weather.md",
      "summary": "서울 날씨 조회",
      "tags": ["weather", "seoul"],
      "created_at": "2026-02-07T10:00:00",
      "turn_count": 4
    }
  ],
  "last_updated": "2026-02-07T10:30:00"
}
```

---

## 📁 구현 파일 구조

```
src/
├── memory/
│   ├── __init__.py
│   ├── session.py          # 세션 저장/로드
│   ├── index.py            # 인덱스 관리
│   ├── search.py           # 메모리 검색 (v2)
│   └── manager.py          # 메모리 매니저
└── tools/
    └── executor.py         # memory_search, memory_save 도구 추가
```

---

## 🔧 핵심 클래스

### 1. `Session` (세션 데이터 클래스)
```python
@dataclass
class Session:
    id: str
    file_path: str
    created_at: datetime
    updated_at: datetime
    summary: str
    tags: list[str]
    messages: list[Message]  # [{"role": "user/assistant", "content": "..."}]
```

### 2. `MemoryManager` (메모리 관리자)
```python
class MemoryManager:
    def __init__(self, memory_dir: str = "memory"):
        ...
    
    def save_session(self, messages: list[dict], summary: str = None) -> str:
        """대화 기록을 세션 파일로 저장"""
    
    def load_session(self, session_id: str) -> Session:
        """세션 ID로 대화 기록 로드"""
    
    def search(self, query: str, limit: int = 5) -> list[Session]:
        """키워드 기반 세션 검색"""
    
    def get_recent(self, limit: int = 3) -> list[Session]:
        """최근 세션 조회"""
    
    def get_context(self, session_id: str = None) -> str:
        """현재/이전 세션 컨텍스트 조합"""
```

---

## 📦 새로운 도구

### `memory_save`
```python
{
    "name": "memory_save",
    "description": "현재 대화를 메모리에 저장",
    "parameters": {
        "summary": "대화 요약 (선택)",
        "tags": "태그 목록 (선택)"
    }
}
```

### `memory_search`
```python
{
    "name": "memory_search", 
    "description": "이전 대화에서 관련 정보 검색",
    "parameters": {
        "query": "검색 키워드",
        "limit": "최대 결과 수 (기본: 5)"
    }
}
```

### `memory_get`
```python
{
    "name": "memory_get",
    "description": "특정 세션의 전체 대화 내용 조회",
    "parameters": {
        "session_id": "세션 ID"
    }
}
```

---

## 🚀 구현 단계

### Phase 1: 기본 저장/로드 (v0.1)
- [ ] `Session` 데이터클래스 정의
- [ ] `MemoryManager.save_session()` 구현
- [ ] `MemoryManager.load_session()` 구현
- [ ] Markdown 파싱/생성 유틸리티
- [ ] 자동 저장 (세션 종료 시)

### Phase 2: 인덱스/검색 (v0.2)
- [ ] `index.json` 관리
- [ ] 키워드 기반 검색 (`search()`)
- [ ] 최근 세션 조회 (`get_recent()`)
- [ ] `memory_search` 도구 추가

### Phase 3: 컨텍스트 통합 (v0.3)
- [ ] 이전 세션 컨텍스트 자동 주입
- [ ] Brain에 "기억하기" 지시 프롬프트 추가
- [ ] 연속 대화 지원

### Phase 4 (미래): 벡터 검색
- [ ] Sentence Transformer 임베딩
- [ ] ChromaDB 통합 (기존 RAG 재사용)
- [ ] 하이브리드 검색

---

## 🎯 예상 사용 흐름

```
사용자: 어제 물어본 날씨 질문 다시 알려줘

시스템:
1. memory_search("날씨") 호출
2. 이전 세션 검색 결과 반환
3. 컨텍스트에 이전 대화 추가
4. Brain이 이전 정보 참조하여 응답

사용자: 이 대화 저장해줘

시스템:
1. memory_save(summary="날씨 조회") 호출
2. 현재 대화를 Markdown으로 저장
3. 인덱스 업데이트
```

---

## ⚠️ 고려사항

1. **파일 크기 관리**: 세션당 최대 턴 수 제한 (100턴)
2. **인덱스 동기화**: 파일 삭제/수정 시 인덱스 자동 갱신
3. **개인정보**: 민감 정보 저장 시 주의 필요
4. **멀티세션**: 동시 실행 시 파일 충돌 방지

---

## 📅 일정

| 단계 | 예상 기간 | 우선순위 |
|------|----------|---------|
| Phase 1 | 2-3시간 | 🔴 높음 |
| Phase 2 | 2-3시간 | 🟡 중간 |
| Phase 3 | 1-2시간 | 🟡 중간 |
| Phase 4 | 4-6시간 | 🟢 낮음 |

---

*작성일: 2026-02-07*
