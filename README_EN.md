[🇰🇷 한국어](README.md) | **🇺🇸 English**

# 🤖 Tiny MoA v2.1 (Unified Agentic System)

> **"AI Legion for the GPU Poor"** - A 1.2B Thinking Model self-plans and orchestrates a 600M Reasoner + 90M Tool Caller to solve complex tasks. ✨

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-0.9+-purple.svg)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/Status-PoC-yellow.svg)]()

![Tiny MoA Demo](docs/img/tiny-moa-demo.gif)

---

## ✨ Key Features

- 🧠 **Multi-Agent & Thinking**: LFM2.5-1.2B-Thinking (Brain) creates plans, collaborating with Reasoner (600M) and Tool Caller (90M).
- 🖥️ **Interactive TUI**: Rich-based real-time task board visualizing inter-agent collaboration.
- 🔧 **Advanced Tooling**: Weather, Search (DuckDuckGo), File RAG, System Control, and more.
- 🌐 **English-First Strategy**: Reasons in English and translates to the user's language for speed and accuracy.
- ⚡ **GPU-Free**: Runs smoothly on 16GB RAM CPU environments.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [How to Run](#-how-to-run)
- [Model Composition](#-model-composition)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/gyunggyung/Tiny-MoA.git
cd Tiny-MoA
```

### 2. Install uv (Recommended)

```powershell
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 3. Install Dependencies

```bash
# Setup check with uv (Recommended - Fast!)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 4. Download Models

```bash
# Brain (LFM2.5-1.2B-Thinking) - *New in v2.1*
huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models/brain

# Reasoner (Falcon-R-0.6B)
huggingface-cli download tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models/reasoner
```

---

## 🏃 How to Run

### Using uv (Recommended)

```bash
# 1. Basic Run (TUI Mode + Thinking)
uv run python -m tiny_moa.main --thinking --show-thinking --tui --query "Compare the weather in Seoul and Tokyo"

# 2. Interactive Mode
uv run python -m tiny_moa.main --interactive

# 3. Long Context Parsing (For complex reports)
uv run python -m tiny_moa.main --thinking --tui --n-ctx 12288 --query "..."

# 4. File Reference (RAG)
uv run python -m tiny_moa.main --tui --query "@[1706.03762v7-split.pdf] What is the main idea of this paper?"

# 5. Web Search (News/Info)
uv run python -m tiny_moa.main --tui --query "Find the latest AI news"

```

### Using pip

```bash
# PYTHONPATH setup required
$env:PYTHONPATH = "src"
python -m tiny_moa.main --query "How is the weather in Seoul?"
```

### Execution Example

```
📝 Input: How is the weather in Seoul?
🌐 Translation: ko → en
🧠 Routing: TOOL
🔧 Executing get_weather
╭──────── 🔧 get_weather Result ───────╮
│ temperature: -2°C                    │
│ condition: Light snow                │
│ humidity: 63%                        │
╰──────────────────────────────────────╯
🌐 Translation: en → ko
💬 Response: The weather in Seoul is -2°C with light snow.
```

---

## 🧩 Model Composition

| Role | Model | Parameters | Memory |
|------|------|----------|--------|
| 🧠 **Brain** | LFM2.5-1.2B-Thinking | 1.17B | ~0.8GB |
| 🤔 **Reasoner** | Falcon-H1-Tiny-R-0.6B | 600M | ~0.4GB |
| 🔧 **Tool Caller** | Falcon-Tool-Calling-90M | 90M | ~0.1GB |

> **Total Memory**: ~2GB (CPU-Only, runs smoothly on 16GB RAM)

---

## 🏗️ Architecture

```
User Input (Multilingual)
       │
       ▼
┌─────────────────────────────────────────┐
│      🌐 Translation Pipeline            │
│  - Language Detect (KR, JP, CN, etc.)   │
│  - Translate to English                 │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│      🧠 Brain (LFM2.5-1.2B)             │
│  - Intent Analysis                      │
│  - Routing: TOOL / REASONER / DIRECT    │
└─────────────────────────────────────────┘
       │
    ┌──┴──────────────┬──────────────┐
    ▼                 ▼              ▼
┌─────────┐     ┌──────────┐   ┌──────────┐
│  TOOL   │     │ REASONER │   │  DIRECT  │
│ Weather/ │     │ Code/Math │   │ Chat     │
│ Search  │     │          │   │          │
└─────────┘     └──────────┘   └──────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│      🌐 Response Translation             │
│  - English → Original Language          │
└─────────────────────────────────────────┘
       │
       ▼
   Final Response (Original Language)
```

---

## 📂 Project Structure

```
Tiny-MoA/
├── pyproject.toml          # uv project configuration
├── uv.lock
├── requirements.txt
├── README.md
├── README_EN.md
├── LICENSE
├── docs/                   # Documentation & Plans
├── models/                 # GGUF Models (Brain, Reasoner)
├── rag_storage/            # RAG Vector DB (ChromaDB)
└── src/
    ├── doc_processing/     # Document Conversion (Docling)
    │   └── converter.py
    ├── rag/                # RAG Engine
    │   ├── engine.py       # RAG Logic
    │   └── store.py        # Vector Store
    ├── tiny_moa/           # Main Package
    │   ├── cowork/         # Tiny Cowork (Agentic Workflow)
    │   │   ├── workers/    # Specialized Workers (Brain, Tool, etc.)
    │   │   ├── planner.py  # Task Planner
    │   │   └── workspace.py# File System Access
    │   ├── ui/             # TUI (Rich)
    │   ├── brain.py        # Thinking Model Wrapper
    │   ├── reasoner.py     # Falcon Wrapper
    │   ├── orchestrator.py # Central Controller
    │   └── main.py         # Entry Point
    ├── tools/              # Tool Use
    │   ├── executor.py     # Tool Executor (Search, Weather, etc.)
    │   └── schema.py       # Tool Definitions
    └── translation/        # Translation Pipeline
```

---

## 📅 Roadmap

- [x] **Phase 0:** Model Research & Architecture Design
- [x] **Phase 1:** Basic Brain + Reasoner Implementation
- [x] **Phase 2:** Tool Calling (Weather, Search, Calc, Time)
- [x] **Phase 3:** Translation Pipeline (English-First Strategy)
- [x] **Phase 4:** TUI & Thinking Model Integration (v2.1)
- [x] **Phase 5:** Docling Document Conversion
- [ ] **Phase 5:** [Agent Ecosystem](docs/agent_ecosystem_vision.md) Construction
- [ ] **Phase 6:** [All-in-One GUI App](docs/tiny_cowork_app_vision.md) Development
- [ ] **Phase 7:** [Master Roadmap](docs/v2_1_master_roadmap.md) Achievement

---

## 📚 References

| Model | Link |
|------|------|
| LFM2.5-1.2B-Instruct | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| LFM2.5-1.2B-Thinking | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| Falcon-H1-Tiny-R-0.6B | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-0.6B) |
| Falcon-Tool-Calling | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) |

---

## 📄 License

This project is distributed under the **Apache 2.0** License.

---

## 📬 Contact

- **Author:** [gyunggyung](https://github.com/gyunggyung)
- **Issues:** [GitHub Issues](https://github.com/gyunggyung/Tiny-MoA/issues)

---

<p align="center">
  <b>🚀 Even the GPU Poor can enjoy AI! 🚀</b>
</p>
