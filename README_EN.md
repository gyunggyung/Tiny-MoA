[🇰🇷 한국어](README.md) | **🇺🇸 English**

# 🤖 Tiny MoA v2.1 (Unified Agentic System)

> **"AI Legion for the GPU Poor"** - 1.2B Thinking Model autonomously plans and executes complex tasks! ✨

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-0.9+-purple.svg)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/Status-PoC-yellow.svg)]()

---

## ✨ Key Features

- 🧠 **Multi-Agent & Thinking**: LFM2.5-1.2B-Thinking (Brain) plans, collaborating with Reasoner (600M) & Tool Caller (90M).
- 🖥️ **Interactive TUI**: Rich-based real-time task dashboard visualizing collaboration process.
- 🔧 **Advanced Tooling**: Weather, Search (DuckDuckGo), File RAG, System Control.
- 🌐 **English-First Strategy**: Reason in English, Translate to Local Language for speed & accuracy.
- ⚡ **GPU-Free**: Runs smoothly on 16GB RAM CPU.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [How to Run](#-how-to-run)
- [Model Configuration](#-model-configuration)
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
# Using uv (recommended - fast!)
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
# 1. Basic Run (TUI + Thinking)
uv run python -m tiny_moa.main --thinking --show-thinking --tui --query "Compare weather in Seoul and Tokyo"

# 2. Interactive Mode
uv run python -m tiny_moa.main --interactive

# 3. Long Context (Complex Reports)
uv run python -m tiny_moa.main --thinking --tui --n-ctx 12288 --query "..."

# 4. File Reference (RAG)
uv run python -m tiny_moa.main --thinking --tui --query "@[1706.03762v7-split.pdf] What is the main idea of this paper?"

# 5. Web Search (News/Info)
uv run python -m tiny_moa.main --thinking --tui --query "Find the latest AI news"
```

### Using pip environment

```bash
# PYTHONPATH setup required
$env:PYTHONPATH = "src"
python -m tiny_moa.main --query "How is the weather in Seoul?"
```

### Example Output

```
📝 Input: How is the weather in Seoul?
🌐 Translation: ko → en
🧠 Routing: TOOL
🔧 get_weather executed
╭──────── 🔧 get_weather result ────────╮
│ temperature: -2°C                      │
│ condition: Light snow                  │
│ humidity: 63%                          │
╰────────────────────────────────────────╯
💬 Response: Seoul weather is -2°C with light snow.
```

---

## 🧩 Model Configuration

| Role | Model | Parameters | Memory |
|------|-------|------------|--------|
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
│  - Language detection (KO, JA, ZH...)   │
│  - Translate to English                 │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│      🧠 Brain (LFM2.5-1.2B)             │
│  - Intent analysis                      │
│  - Routing: TOOL / REASONER / DIRECT    │
└─────────────────────────────────────────┘
       │
    ┌──┴──────────────┬──────────────┐
    ▼                 ▼              ▼
┌─────────┐     ┌──────────┐   ┌──────────┐
│  TOOL   │     │ REASONER │   │  DIRECT  │
│Weather  │     │Code/Math │   │  Chat    │
└─────────┘     └──────────┘   └──────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│      🌐 Response Translation            │
│  - English → Original language          │
└─────────────────────────────────────────┘
       │
       ▼
   Final Response (Original Language)
```

---

## 📂 Project Structure

```
Tiny-MoA/
├── pyproject.toml          # uv project config
├── uv.lock                 # Dependency lock file
├── requirements.txt        # pip compatible
├── README.md
├── README_EN.md
├── docs/
│   ├── implementation_plan.md
│   ├── tool_calling_plan.md
│   └── translation_multiagent_plan.md
├── models/                 # GGUF models (gitignored)
│   ├── brain/
│   └── reasoner/
└── src/
    ├── tiny_moa/           # Main module
    │   ├── brain.py        # Brain model wrapper
    │   ├── reasoner.py     # Reasoner model wrapper
    │   ├── orchestrator.py # Orchestrator
    │   └── main.py         # Entry point
    ├── tools/              # Tool Calling
    │   ├── schema.py       # Tool schema
    │   ├── executor.py     # Tool executor
    │   └── caller.py       # Tool caller
    └── translation/        # Translation module
        ├── detector.py     # Language detection
        ├── translator.py   # Google Translate
        └── pipeline.py     # Translation pipeline
```

---

## 📅 Roadmap

- [x] **Phase 0:** Model research & architecture design
- [x] **Phase 1:** Brain + Reasoner basic implementation
- [x] **Phase 2:** Tool Calling (weather, search, calc, time)
- [x] **Phase 3:** Translation pipeline (English-First Strategy)
- [x] **Phase 4:** TUI & Thinking Model Integration (v2.1)
- [ ] **Phase 5:** [Agent Ecosystem](docs/agent_ecosystem_vision.md)
- [ ] **Phase 6:** [All-in-One GUI App](docs/tiny_cowork_app_vision.md)
- [ ] **Phase 7:** [Master Roadmap](docs/v2_1_master_roadmap.md)

---

## 📚 References

| Model | Link |
|-------|------|
| LFM2.5-1.2B-Instruct | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| LFM2.5-1.2B-Thinking | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| Falcon-H1-Tiny-R-0.6B | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-0.6B) |
| Falcon-Tool-Calling | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) |

---

## 📄 License

This project is licensed under **Apache 2.0**.

---

## 📬 Contact

- **Author:** [gyunggyung](https://github.com/gyunggyung)
- **Issues:** [GitHub Issues](https://github.com/gyunggyung/Tiny-MoA/issues)

---

<p align="center">
  <b>🚀 AI for the GPU Poor! 🚀</b>
</p>
