"""
Language Detector
=================
언어 감지 모듈 - langdetect 또는 간단한 휴리스틱 사용
"""

import re
from typing import Optional

# 언어 코드 매핑
LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
}


def detect_language(text: str) -> str:
    """
    텍스트의 언어를 감지합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        언어 코드 (예: 'ko', 'en', 'ja')
    """
    if not text or not text.strip():
        return "en"
    
    # 1. langdetect 사용 시도 (정확도 높음)
    try:
        from langdetect import detect, DetectorFactory
        # 일관된 결과를 위해 시드 설정
        DetectorFactory.seed = 0
        detected = detect(text)
        # 중국어 통합 (zh-cn, zh-tw -> zh)
        if detected.startswith("zh"):
            return "zh"
        return detected
    except ImportError:
        pass
    except Exception:
        pass
    
    # 2. 휴리스틱 폴백 (langdetect 없을 때)
    return _detect_by_unicode(text)


def _detect_by_unicode(text: str) -> str:
    """
    유니코드 범위 기반 언어 감지 (간단한 휴리스틱)
    """
    # 한글 범위
    korean_pattern = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]')
    # 일본어 범위 (히라가나, 가타카나)
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
    # 중국어 범위 (한자)
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF]')
    # 키릴 문자 (러시아어 등)
    cyrillic_pattern = re.compile(r'[\u0400-\u04FF]')
    # 아랍어
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    # 태국어
    thai_pattern = re.compile(r'[\u0E00-\u0E7F]')
    
    # 각 패턴 매치 수 계산
    counts = {
        "ko": len(korean_pattern.findall(text)),
        "ja": len(japanese_pattern.findall(text)),
        "zh": len(chinese_pattern.findall(text)),
        "ru": len(cyrillic_pattern.findall(text)),
        "ar": len(arabic_pattern.findall(text)),
        "th": len(thai_pattern.findall(text)),
    }
    
    # 가장 많이 매치된 언어 선택
    max_lang = max(counts, key=counts.get)
    if counts[max_lang] > 0:
        # 일본어는 한자도 사용하므로, 히라가나/가타카나가 있으면 일본어 우선
        if counts["ja"] > 0:
            return "ja"
        return max_lang
    
    # 기본값: 영어
    return "en"


def is_english(text: str) -> bool:
    """
    텍스트가 영어인지 확인합니다.
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        영어이면 True
    """
    return detect_language(text) == "en"


def get_language_name(code: str) -> str:
    """
    언어 코드를 이름으로 변환합니다.
    
    Args:
        code: 언어 코드 (예: 'ko')
        
    Returns:
        언어 이름 (예: 'Korean')
    """
    return LANGUAGE_NAMES.get(code, code.upper())


if __name__ == "__main__":
    # 테스트
    test_texts = [
        "Hello, how are you?",
        "안녕하세요, 오늘 날씨가 좋네요.",
        "こんにちは、元気ですか？",
        "你好，今天天气很好。",
        "Bonjour, comment allez-vous?",
        "Привет, как дела?",
    ]
    
    print("=== 언어 감지 테스트 ===")
    for text in test_texts:
        lang = detect_language(text)
        name = get_language_name(lang)
        print(f"[{lang}] {name}: {text[:30]}...")
