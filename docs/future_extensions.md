# 🔮 향후 확장 계획: Docling & 다중 에이전트

> **상태:** 계획 단계 - uv 환경 설정 후 진행

---

## 📋 남은 구현 항목

### 1. Docling 문서 변환 모듈 (`src/docling/`)
- PDF, DOCX, PPTX → Markdown 변환
- @ 멘션 파일 참조 기능
- 청킹 처리 (큰 문서 대응)

### 2. 에이전트 시스템 (`src/agents/`)
- 기본 에이전트 클래스
- 한국 문화 전문 에이전트
- RAG 에이전트

### 3. RAG 시스템 (`src/rag/`)
- 임베딩 모델 (sentence-transformers)
- 벡터 저장소 (ChromaDB)
- 문서 검색기

---

## ⚠️ 선행 작업: uv 환경 설정

위 모듈들은 추가 의존성이 필요하므로, 먼저 uv 환경을 구성한 후 진행합니다.

```bash
# docling 추가 시
uv add docling

# RAG 관련 추가 시  
uv add sentence-transformers chromadb
```

---

## 📅 예상 일정

| 작업 | 예상 시간 |
|------|-----------|
| Docling 통합 | ~2시간 |
| 에이전트 프레임워크 | ~2시간 |
| RAG 시스템 | ~4시간 |

---

자세한 내용은 `translation_multiagent_plan.md` 참조.
