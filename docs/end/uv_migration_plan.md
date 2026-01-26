# ⚡ uv Migration Plan for Tiny MoA

이 문서는 현재 `pip` + `requirements.txt` 기반의 프로젝트를 차세대 Python 패키지 매니저인 [`uv`](https://github.com/astral-sh/uv) 환경으로 전환하기 위한 상세 계획입니다.

## 1. 개요

`uv`는 Rust로 작성된 매우 빠른 Python 패키지 매니저입니다. 이 프로젝트의 "가볍고 효율적인 AI" 철학에 맞춰 개발 환경 또한 가장 가볍고 효율적인 도구로 전환합니다.

### 전환 이점
- **속도**: 의존성 해결 및 설치 속도가 pip 대비 10~100배 빠름
- **표준 준수**: `pyproject.toml` 표준을 사용하여 메타데이터와 의존성을 체계적으로 관리
- **재현성**: `uv.lock` 파일을 통해 모든 개발자(및 환경)에서 비트 단위로 동일한 패키지 버전 보장
- **편의성**: 가상환경(`python -m venv`)을 직접 관리할 필요 없이 `uv`가 프로젝트별로 자동 관리

---

## 2. 사전 준비 (Prerequisites)

### uv 설치
Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 확인:
```bash
uv --version
```

---

## 3. 마이그레이션 절차 (Step-by-Step)

### Step 1: 프로젝트 초기화
기존 프로젝트 루트(`c:\github\MoA-PoC`)에서 실행합니다.

```bash
# uv 프로젝트 초기화 (pyproject.toml 생성)
uv init
```

### Step 2: 의존성 이전
`requirements.txt`에 있는 패키지들을 `uv add` 명령어로 `pyproject.toml`에 등록합니다.

**Core 의존성:**
```bash
uv add llama-cpp-python>=0.3.0
uv add huggingface-hub>=0.25.0
```

**Utilities:**
```bash
uv add rich>=13.0.0
uv add pydantic>=2.0.0
```

> **Note:** `uv add`를 실행하면 자동으로 가상환경이 생성되고 패키지가 설치되며 `uv.lock` 파일이 생성됩니다.

### Step 3: 프로젝트 메타데이터 업데이트 (`pyproject.toml`)
생성된 `pyproject.toml` 파일을 열어 프로젝트 정보를 수정합니다.

```toml
[project]
name = "tiny-moa"
version = "0.1.0"
description = "Tiny MoA (Mixture of Agents) PoC - AI Legion for the GPU Poor"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "huggingface-hub>=0.25.0",
    "llama-cpp-python>=0.3.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
]
```

### Step 4: 정리 (Cleanup)
기존 환경 설정 파일들을 삭제합니다.

```bash
# 더 이상 필요 없는 파일 삭제
rm requirements.txt

# (선택) 기존 가상환경 폴더 삭제
# rm -r .venv (주의: uv도 기본적으로 .venv를 사용하므로, 기존 것을 지우고 uv sync를 하는 것이 깔끔함)
```

---

## 4. 새로운 워크플로우 (How to Work)

전환 후에는 다음과 같이 작업합니다.

### 실행 방법
가상환경을 `activate` 할 필요 없이 `uv run`으로 실행합니다.

```bash
# 기존: python -m tiny_moa.main --query "..."
# 변경:
uv run -m tiny_moa.main --query "Hello"
```

`uv run`은 프로젝트 루트를 자동으로 인식하고 가상환경 내의 파이썬을 사용하므로, `PYTHONPATH` 설정이 필요 없거나 훨씬 간단해집니다.

### 패키지 추가/삭제
```bash
uv add numpy        # 패키지 추가
uv remove numpy     # 패키지 삭제
```

### 환경 동기화 (다른 PC에서 실행 시)
```bash
git clone ...
cd MoA-PoC
uv sync             # uv.lock 기반으로 가상환경 자동 생성 및 패키지 설치
```

---

## 5. 실행 결과 예시 (Anticipated)

```
$ uv run -m tiny_moa.main --query "test"
Resolved 4 packages in 10ms
Prepared 1 package in 20ms
Installed 1 package in 5ms
 + tiny-moa==0.1.0 (from file://...)
... (프로그램 실행 결과) ...
```

---

## 6. 주의사항

- **`llama-cpp-python` 빌드:** Windows에서 `llama-cpp-python` 설치 시 컴파일러 설정이나 CUDA 설정이 필요한 경우, `uv pip install` 방식이나 별도 빌드 플래그가 필요할 수 있습니다. (기본 `uv add`로 휠(wheel) 설치가 되면 문제없음)
- **VSCode 연동:** VSCode가 `uv`가 생성한 `.venv`를 자동으로 인식합니다. 하단 인터프리터 선택에서 `.venv` 안의 python을 선택해주면 됩니다.
