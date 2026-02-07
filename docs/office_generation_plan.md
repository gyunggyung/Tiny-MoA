# 📄 Tiny-MoA Office Document Generation Plan

> **PPT, Word, Excel 문서 자동 생성 기능 구현 계획**  
> 참고: [Reference/Gemini-Claw/src/office](../Reference/Gemini-Claw/src/office/)

---

## 📋 개요

### 목표
- AI가 사용자 요청에 따라 전문적인 Office 문서 자동 생성
- 한국어/다국어 폰트 지원
- 검색/분석 결과를 문서로 자동 변환

### Gemini-Claw 참고 사항
| 기능 | 라이브러리 | 용도 |
|------|-----------|------|
| Word | `python-docx` | 보고서, 문서 생성 |
| PowerPoint | `python-pptx` | 프레젠테이션 생성 |
| Excel | `openpyxl`, `pandas` | 데이터 분석/시각화 |

---

## 🏗️ 아키텍처

```
src/
├── office/
│   ├── __init__.py
│   ├── agent.py           # OfficeAgent 메인 클래스
│   ├── word.py            # Word 문서 생성
│   ├── powerpoint.py      # PPT 생성
│   ├── excel.py           # Excel 처리
│   └── styles.py          # 공통 스타일 (폰트, 색상)
└── tools/
    └── executor.py        # create_word, create_ppt, create_excel 도구 추가
```

---

## 🔧 핵심 클래스

### `OfficeAgent`
```python
class OfficeAgent:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.default_font = "Malgun Gothic"  # 한국어 폰트
    
    def create_word_report(
        self, 
        title: str, 
        sections: list[dict],  # [{"heading": "...", "content": "..."}]
        output_path: str
    ) -> str:
        """구조화된 Word 보고서 생성"""
    
    def create_presentation(
        self,
        title: str,
        subtitle: str,
        slides: list[dict],  # [{"title": "...", "content": [...]}]
        output_path: str
    ) -> str:
        """전문적인 PPT 프레젠테이션 생성"""
    
    def create_excel(
        self,
        data: list[dict],
        output_path: str,
        sheet_name: str = "Data"
    ) -> str:
        """스타일링된 Excel 파일 생성"""
    
    def process_excel(
        self,
        input_path: str,
        output_path: str
    ) -> str:
        """Excel 데이터 분석 및 요약 생성"""
```

---

## 📦 새로운 도구

### `create_word`
```python
{
    "name": "create_word",
    "description": "Word 보고서 문서 생성",
    "parameters": {
        "title": "문서 제목",
        "sections": "섹션 목록 [{heading, content}]",
        "filename": "파일명 (기본: report.docx)"
    }
}
```

### `create_ppt`
```python
{
    "name": "create_ppt",
    "description": "PowerPoint 프레젠테이션 생성",
    "parameters": {
        "title": "발표 제목",
        "subtitle": "부제목",
        "slides": "슬라이드 목록 [{title, content}]",
        "filename": "파일명 (기본: presentation.pptx)"
    }
}
```

### `create_excel`
```python
{
    "name": "create_excel",
    "description": "Excel 스프레드시트 생성",
    "parameters": {
        "data": "데이터 목록 [{col1, col2, ...}]",
        "filename": "파일명 (기본: data.xlsx)"
    }
}
```

---

## 🎨 스타일링 기능

### 공통 스타일
```python
STYLE_CONFIG = {
    "font_kr": "Malgun Gothic",
    "font_en": "Calibri",
    "primary_color": "#366092",
    "accent_color": "#4472C4",
    "header_bg": "#366092",
    "header_fg": "#FFFFFF",
}
```

### Word 스타일
- 제목: 가운데 정렬, 굵게
- 섹션 헤딩: Level 1, 파란색
- 본문: 줄간격 1.5, 한국어 폰트 적용

### PPT 스타일
- 타이틀 슬라이드: 중앙 정렬
- 콘텐츠 슬라이드: 불릿 포인트
- 한국어 폰트 자동 적용

### Excel 스타일
- 헤더: 배경색, 굵은 폰트, 가운데 정렬
- 자동 열 너비 조정
- 교차 행 색상 (선택)

---

## 🚀 구현 단계

### Phase 1: 기본 구조 (v0.1)
- [ ] `OfficeAgent` 클래스 생성
- [ ] 의존성 추가 (`python-docx`, `python-pptx`, `openpyxl`)
- [ ] Word 보고서 기본 생성
- [ ] PPT 프레젠테이션 기본 생성

### Phase 2: 스타일링 (v0.2)
- [ ] 한국어 폰트 지원
- [ ] 전문적인 스타일 (색상, 정렬)
- [ ] Excel 헤더/열 너비 자동 조정

### Phase 3: 도구 통합 (v0.3)
- [ ] `create_word`, `create_ppt`, `create_excel` 도구 추가
- [ ] Brain 프롬프트에 Office 기능 안내 추가
- [ ] Cowork 결과 → 문서 자동 변환

### Phase 4: 고급 기능 (v0.4)
- [ ] 템플릿 기반 문서 생성
- [ ] 차트/그래프 삽입 (Excel → PPT)
- [ ] 이미지 삽입 지원

---

## 📝 예상 사용 흐름

```
사용자: OpenAI와 Anthropic 최신 뉴스 분석해서 PPT로 만들어줘

시스템:
1. search_news("OpenAI latest news") 실행
2. search_news("Anthropic latest news") 실행
3. Brain이 뉴스 분석 및 정리
4. create_ppt() 호출
   - 슬라이드 1: 제목
   - 슬라이드 2: OpenAI 동향
   - 슬라이드 3: Anthropic 동향
   - 슬라이드 4: 비교 분석
5. 결과 파일 반환: output/ai_news_analysis.pptx
```

---

## 📦 의존성

```toml
# pyproject.toml에 추가
[project.dependencies]
python-docx = ">=1.1.0"
python-pptx = ">=0.6.23"
openpyxl = ">=3.1.0"
pandas = ">=2.0.0"
```

---

## ⚠️ 고려사항

1. **언어 일치**: 사용자 질문 언어와 문서 내용 언어 일치
2. **파일명 규칙**: 파일명은 영어/ASCII만 사용 (인코딩 문제 방지)
3. **출력 디렉토리**: `output/` 폴더에 저장
4. **파일 덮어쓰기**: 기존 파일 존재 시 타임스탬프 추가

---

## 📅 일정

| 단계 | 예상 기간 | 우선순위 |
|------|----------|---------|
| Phase 1 | 2-3시간 | 🔴 높음 |
| Phase 2 | 1-2시간 | 🟡 중간 |
| Phase 3 | 2-3시간 | 🟡 중간 |
| Phase 4 | 4-6시간 | 🟢 낮음 |

---

*작성일: 2026-02-07*
