**🇰🇷 한국어** | [🇺🇸 English](README_EN.md)

# 🤖 Tiny MoA v2.1 (Unified Agentic System)

> **"GPU Poor를 위한 AI 군단"** - 1.2B Thinking Model이 스스로 계획하고 600M Reasoner + 90M Tool Caller 조합으로 복잡한 작업을 수행합니다. ✨

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-0.9+-purple.svg)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/Status-PoC-yellow.svg)]()

---

## ✨ 주요 기능

- 🧠 **Multi-Agent & Thinking**: LFM2.5-1.2B-Thinking (Brain)이 계획을 수립하고, Reasoner(600M)와 Tool Caller(90M)가 협업.
- 🖥️ **Interactive TUI**: Rich 기반의 실시간 태스크 보드로 에이전트 간 협업 과정 시각화.
- 🔧 **Advanced Tooling**: 날씨, 검색(DuckDuckGo), 파일 RAG, 시스템 제어 등 강력한 도구 연동.
- 🌐 **English-First Strategy**: 영어로 추론하고 한국어로 번역하여 속도와 정확도 동시 확보.
- ⚡ **GPU-Free**: 16GB RAM CPU 환경에서도 쾌적한 구동.

---

## 📋 목차

- [빠른 시작](#-빠른-시작)
- [실행 방법](#-실행-방법)
- [모델 구성](#-모델-구성)
- [아키텍처](#-아키텍처)
- [프로젝트 구조](#-프로젝트-구조)
- [로드맵](#-로드맵)

---

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/gyunggyung/Tiny-MoA.git
cd Tiny-MoA
```

### 2. uv 설치 (권장)

```powershell
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 설치 확인
uv --version
```

### 3. 의존성 설치

```bash
# uv로 환경 설정 (권장 - 빠름!)
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### 4. 모델 다운로드

```bash
# Brain (LFM2.5-1.2B-Thinking) - *New in v2.1*
huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models/brain

# Reasoner (Falcon-R-0.6B)
huggingface-cli download tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models/reasoner
```

---

## 🏃 실행 방법

### uv 사용 (권장)

```bash
# 1. 기본 실행 (TUI 모드 + Thinking)
uv run python -m tiny_moa.main --thinking --show-thinking --tui --query "서울과 도쿄 날씨 비교해줘"

# 2. 대화형 모드
uv run python -m tiny_moa.main --interactive

# 3. 긴 문맥 처리 (복잡한 리포트 생성 시)
uv run python -m tiny_moa.main --thinking --tui --n-ctx 12288 --query "..."

# 4. 파일 참조 (RAG)
uv run python -m tiny_moa.main --tui --query "@[1706.03762v7-split.pdf] 이 논문의 주요 아이디어가 뭐야?"

# 5. 웹 검색 (뉴스/정보)
uv run python -m tiny_moa.main --tui --query "최신 AI 뉴스 찾아줘"

```

### pip 환경 사용

```bash
# PYTHONPATH 설정 필요
$env:PYTHONPATH = "src"
python -m tiny_moa.main --query "서울 날씨 어때?"
```

### 실행 예시

```
📝 입력: 서울 날씨 어때?
🌐 번역: ko → en
🧠 라우팅: TOOL
🔧 get_weather 실행
╭──────── 🔧 get_weather 결과 ────────╮
│ temperature: -2°C                   │
│ condition: Light snow               │
│ humidity: 63%                       │
╰─────────────────────────────────────╯
🌐 번역: en → ko
💬 응답: 서울 날씨는 -2°C이고 가벼운 눈이 내리고 있습니다.
```

---

## 🧩 모델 구성

| 역할 | 모델 | 파라미터 | 메모리 |
|------|------|----------|--------|
| 🧠 **Brain** | LFM2.5-1.2B-Thinking | 1.17B | ~0.8GB |
| 🤔 **Reasoner** | Falcon-H1-Tiny-R-0.6B | 600M | ~0.4GB |
| 🔧 **Tool Caller** | Falcon-Tool-Calling-90M | 90M | ~0.1GB |

> **총 메모리**: ~2GB (CPU-Only, 16GB RAM에서 원활히 구동)

---

## 🏗️ 아키텍처

```
사용자 입력 (다국어)
       │
       ▼
┌─────────────────────────────────────────┐
│      🌐 번역 파이프라인                  │
│  - 언어 감지 (한국어, 일본어, 중국어)    │
│  - 영어로 번역                          │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│      🧠 Brain (LFM2.5-1.2B)             │
│  - 의도 분석                            │
│  - 라우팅: TOOL / REASONER / DIRECT      │
└─────────────────────────────────────────┘
       │
    ┌──┴──────────────┬──────────────┐
    ▼                 ▼              ▼
┌─────────┐     ┌──────────┐   ┌──────────┐
│  TOOL   │     │ REASONER │   │  DIRECT  │
│ 날씨/검색│     │ 코딩/수학 │   │ 일반대화 │
└─────────┘     └──────────┘   └──────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│      🌐 응답 번역                        │
│  - 영어 → 원래 언어로 변환              │
└─────────────────────────────────────────┘
       │
       ▼
   최종 응답 (원래 언어)
```

---

## 📂 프로젝트 구조

```
Tiny-MoA/
├── pyproject.toml          # uv 프로젝트 설정
├── uv.lock
├── requirements.txt
├── README.md
├── README_EN.md
├── LICENSE
├── docs/                   # 문서 및 계획 (Plans, Roadmaps)
├── models/                 # GGUF 모델 (Brain, Reasoner)
├── rag_storage/            # RAG 벡터 DB (ChromaDB)
└── src/
    ├── doc_processing/     # 문서 변환 (Docling)
    │   └── converter.py
    ├── rag/                # RAG 엔진
    │   ├── engine.py       # RAG 로직
    │   └── store.py        # 벡터 저장소
    ├── tiny_moa/           # 메인 패키지
    │   ├── cowork/         # Tiny Cowork (Agentic Workflow)
    │   │   ├── workers/    # Specialized Workers (Brain, Tool, etc.)
    │   │   ├── planner.py  # 작업 계획
    │   │   └── workspace.py# 파일 시스템 접근
    │   ├── ui/             # TUI (Rich)
    │   ├── brain.py        # Thinking Model Wrapper
    │   ├── reasoner.py     # Falcon Wrapper
    │   ├── orchestrator.py # Central Controller
    │   └── main.py         # Entry Point
    ├── tools/              # Tool Use
    │   ├── executor.py     # 도구 실행 (Search, Weather, etc.)
    │   └── schema.py       # 도구 정의
    └── translation/        # 번역 파이프라인
```

---

## 📅 로드맵

- [x] **Phase 0:** 모델 연구 및 아키텍처 설계
- [x] **Phase 1:** Brain + Reasoner 기본 구현
- [x] **Phase 2:** Tool Calling (날씨, 검색, 계산, 시간)
- [x] **Phase 3:** 번역 파이프라인 (English-First Strategy 적용)
- [x] **Phase 4:** TUI 및 Thinking Model 통합 (v2.1)
- [x] **Phase 5:** Docling 문서 변환
- [ ] **Phase 5:** [Agent Ecosystem](docs/agent_ecosystem_vision.md) 구축
- [ ] **Phase 6:** [All-in-One GUI App](docs/tiny_cowork_app_vision.md) 개발
- [ ] **Phase 7:** [Master Roadmap](docs/v2_1_master_roadmap.md) 달성

---

## 📚 참고 자료

| 모델 | 링크 |
|------|------|
| LFM2.5-1.2B-Instruct | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| LFM2.5-1.2B-Thinking | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| Falcon-H1-Tiny-R-0.6B | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-0.6B) |
| Falcon-Tool-Calling | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) |

---

## 📄 라이선스

이 프로젝트는 **Apache 2.0** 라이선스로 배포됩니다.

---

## 📬 연락처

- **작성자:** [gyunggyung](https://github.com/gyunggyung)
- **이슈:** [GitHub Issues](https://github.com/gyunggyung/Tiny-MoA/issues)

---

<p align="center">
  <b>🚀 GPU Poor도 AI를 누릴 수 있다! 🚀</b>
</p>
