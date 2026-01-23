[🇰🇷 한국어](README.md) | **🇺🇸 English**

# 🤖 Tiny MoA (Mixture of Agents) PoC

> **"AI Legion for the GPU Poor"** - Better performance with a 1.2B Brain + 600M Specialist combo instead of a single 4B model!

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-PoC-yellow.svg)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Ideas](#-key-ideas)
- [Model Configuration](#-model-configuration)
- [Memory Requirements](#-memory-requirements)
- [Quick Start](#-quick-start)
- [How to Run](#-how-to-run)
- [Architecture](#-architecture)
- [Examples](#-examples)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🎯 Overview

**Tiny MoA** is a **Mixture of Agents** architecture built by combining the latest "Small but Powerful" models of 2026.

### Why Tiny MoA?

| Traditional Method | Tiny MoA |
|--------------------|----------|
| Uses single 4B~7B model | 1.2B Brain + 600M Specialist |
| Requires 8~16GB VRAM | **2GB RAM is enough (CPU-Only)** |
| Average performance of single model | **Experts optimized for specific fields** |
| Heavy loading time | Instant response with lightweight models |

---

## 💡 Key Ideas

```
┌─────────────────────────────────────────────────────────────┐
│                      🧠 Brain (1.2B)                        │
│              LiquidAI LFM2.5 (Thinking/Instruct)            │
│       Intent Analysis → Task Decomposition → Select Expert  │
│                   → Result Integration                      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    🔧 Specialist (600M)                     │
│              Falcon-H1-Tiny-R-0.6B (Coding+Math)            │
│                  LiveCodeBench 39% + MATH500 94%            │
└─────────────────────────────────────────────────────────────┘
```

The **Brain** analyzes the user's request and delegates tasks by calling the **Specialist**.

---

## 🧩 Model Configuration (PoC)

### 🧠 Brain (Central Commander)

| Model | Parameters | Role | Quantization |
|-------|------------|------|--------------|
| **LFM2.5-1.2B** | 1.17B | Router, Integration, Korean support | Q4_K_M (~0.8GB) |

> **⚠️ Thinking vs Instruct:** To be decided after PoC experiments.

**Key Features:**
- Trained on 28T tokens
- Supports 8 languages (including Korean) → **No need for separate Korean model**
- IFEval 86.23%, BFCLv3 49.12%
- Recommended parameters: `temperature=0.7, top_p=0.9, repeat_penalty=1.1` (Official llama.cpp docs)

### 🔧 Specialist (Expert)

| Role | Model | Parameters | Quantization | Features |
|------|-------|------------|--------------|----------|
| **Coding+Math** | Falcon-H1-Tiny-R-0.6B | 600M | Q4_K_M (~0.4GB) | LiveCodeBench 39%, MATH500 94% |
| Tool Calling (Optional) | Falcon-H1-Tiny-Tool-Calling | 90M | Q8_0 (~0.1GB) | Relevance judgment 94.44% |

**Design Rationale:**
- Falcon-R-0.6B integrates **both Coding + Math roles**
- LFM2.5 handles Korean directly → No need for Multilingual specialists

---

## 💾 Memory Requirements

> **🖥️ PoC Target Environment:** Windows CPU-Only, 16GB RAM, Intel Core i5

### PoC Recommended Configuration (Total ~2.0GB)

```
Brain:     LFM2.5-1.2B (Q4_K_M)        ~0.8GB
Reasoner:  Falcon-R-0.6B (Q4_K_M)      ~0.4GB  ← Coding+Math Integrated!
Tool:      Falcon-Tool-Calling (Q8_0)  ~0.1GB  (Optional)
KV Cache + OS Overlay                  ~0.7GB
─────────────────────────────────────────
Total                                  ~2.0GB
```

### Dynamic Loading Strategy

```python
# Always Loaded (PoC Core)
always_loaded = ["brain", "reasoner"]

# Load on Demand
on_demand = ["tool_caller"]

# Just-In-Time (For extension)
just_in_time = ["ocr", "audio"]
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# 1. Clone repository
git clone https://github.com/gyunggyung/MoA-PoC.git
cd MoA-PoC

# 2. Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download GGUF models for llama.cpp
pip install huggingface-hub

# Brain (Q4_K_M recommended - CPU environment)
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct-GGUF ^
    --include "*Q4_K_M.gguf" --local-dir ./models/brain

# Reasoner (Coding+Math Integrated)
huggingface-cli download tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF ^
    --include "*Q4_K_M.gguf" --local-dir ./models/reasoner

# Tool Caller (Optional)
huggingface-cli download tiiuae/Falcon-H1-Tiny-Tool-Calling-GGUF ^
    --include "*Q8_0.gguf" --local-dir ./models/tool
```

---

## 🏃 How to Run

### Basic Execution

```bash
# Set PYTHONPATH and run
$env:PYTHONPATH = "src"

# Run single query (Ask in English - Falcon only supports English)
python -m tiny_moa.main --query "Write a Python function for Fibonacci"

# Interactive Mode
python -m tiny_moa.main --interactive
```

### Usage in Python

```python
from tiny_moa import TinyMoA

# Initialize
moa = TinyMoA()

# Coding request → Processed by Reasoner (Ask in English)
response = moa.chat("Write a Python function for Fibonacci sequence")
print(response)

# Math request → Processed by Reasoner
response = moa.chat("What is 1 + 1?")
print(response)

# General chat → Processed by Brain directly (Korean available)
response = moa.chat("Hello!")
print(response)
```

> **⚠️ Note:** Falcon-R-0.6B supports **English only**. Please ask coding/math questions in English.

### Individual Model Test

```bash
# Test Brain only
python -m tiny_moa.brain

# Test Reasoner only
python -m tiny_moa.reasoner
```

---

## 🏗️ Architecture

### System Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    🧠 Brain (LFM2.5-1.2B)                │
│  1. Intent Analysis: "Coding request? Math? General?"    │
│  2. Routing Decision:                                    │
│     - Coding/Math → Call REASONER                        │
│     - General/Korean → Brain processes directly          │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              🤔 Reasoner (Falcon-R-0.6B)                 │
│  - Coding: "Write a Python function for Fibonacci"      │
│  - Math: "Solve step by step..."                        │
│  - Output: Code or Solution details                     │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    🧠 Brain (Integration)                │
│  - Collect Specialist output                             │
│  - Final response formatting (Translation etc.)          │
└────────────────────────┬────────────────────────────────┘
                         ▼
                    Final Response
```

### Routing Rules

| Keyword/Pattern | Handling |
|-----------------|----------|
| code, function, implementation, Python, algorithm | `REASONER` |
| calculate, prove, math, AIME, problem | `REASONER` |
| hello, explain, translate, Korean | `BRAIN Direct` |

---

## 📝 Examples

### Example 1: Coding Request

```python
>>> moa.chat("Implement Quicksort algorithm in Python")

# Brain Analysis: Coding Request → Call REASONER
# REASONER Output:
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

### Example 2: Math Problem

```python
>>> moa.chat("AIME Style: Let n be a positive integer...")

# Brain Analysis: Math Reasoning → Call REASONER
# REASONER (600M): Step-by-step resolution with Chain-of-Thought
# Result: Detailed solution process and answer
```

### Example 3: General Conversation (Korean)

```python
>>> moa.chat("오늘 날씨가 좋네요. 뭘 하면 좋을까요?")
# (The weather is nice today. What should I do?)

# Brain Analysis: General Conversation → Brain processes directly
# LFM2.5 responds directly in Korean
```

---

## 📅 Roadmap

- [x] **Phase 0:** Model Research and Architecture Design
- [ ] **Phase 1:** Brain + Reasoner Basic Implementation ← **Current**
- [ ] **Phase 2:** Tool-Calling Integration
- [ ] **Phase 3:** OCR/docling Integration (Extension)
- [ ] **Phase 4:** Benchmark and Optimization

---

## 📂 Project Structure

```
MoA-PoC/
├── README.md               # Korean README
├── README_EN.md            # English README
├── requirements.txt
├── docs/
│   └── implementation_plan.md  # Detailed Specifications
├── models/                 # GGUF model storage (gitignore)
│   ├── brain/
│   ├── reasoner/
│   └── tool/
├── src/
│   └── tiny_moa/
│       ├── __init__.py
│       ├── main.py             # Entry Point
│       ├── brain.py            # Brain Model Wrapper
│       ├── reasoner.py         # Reasoner Model Wrapper
│       ├── router.py           # Routing Logic
│       └── orchestrator.py     # Orchestrator
└── tests/
    └── test_basic.py
```

---

## 📚 References

### Model Sources and Recommended Parameters

| Model | Link | Recommended Parameters |
|-------|------|------------------------|
| LFM2.5-1.2B-Instruct | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | `temp=0.1, top_k=50, rep_penalty=1.05` |
| LFM2.5-1.2B-Thinking | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) | `temp=0.05, top_k=50, rep_penalty=1.05` |
| Falcon-H1-Tiny-R-0.6B | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-0.6B) | `temp=0.1` |
| Falcon-Tool-Calling | [HuggingFace](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) | `temp=0.1` |

### Blogs and Technical Docs

- [Falcon-H1-Tiny Blog Post](https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost)
- [Introducing LiquidAI LFM2.5](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)

---

## 📄 License

This project is distributed under the Apache 2.0 License.

**Model Licenses:**
- LFM2.5: [Liquid AI License](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
- Falcon-H1-Tiny: [Falcon-LLM License](https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct)

---

## 📬 Contact

- **Author:** [gyunggyung](https://github.com/gyunggyung)
- **Issues:** [GitHub Issues](https://github.com/gyunggyung/MoA-PoC/issues)

---

<p align="center">
  <b>🚀 AI for the GPU Poor! 🚀</b>
</p>
