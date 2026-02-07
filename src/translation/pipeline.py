"""
Translation Pipeline
====================
번역 파이프라인 - 언어 감지 → 영어 번역 → 모델 처리 → 원래 언어 번역
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable

from .detector import detect_language, is_english
from .translator import create_translator

logger = logging.getLogger(__name__)


@dataclass
class TranslationContext:
    """번역 컨텍스트 - 원본 언어 정보 보존"""
    original_text: str
    original_lang: str
    english_text: str
    is_translated: bool


class TranslationPipeline:
    """
    번역 파이프라인
    
    다국어 입력 → 영어로 번역 → 모델 처리 → 원래 언어로 번역
    
    사용 예시:
        pipeline = TranslationPipeline()
        
        # 1. 입력을 영어로 변환
        ctx = pipeline.to_english("안녕하세요, 오늘 날씨 어때요?")
        print(ctx.english_text)  # "Hello, how is the weather today?"
        
        # 2. 모델 처리 (영어로)
        model_response = model.generate(ctx.english_text)
        
        # 3. 응답을 원래 언어로 변환
        final_response = pipeline.from_english(model_response, ctx)
        print(final_response)  # "안녕하세요, 오늘 날씨가 좋습니다."
    """
    
    def __init__(self, use_simple_translator: bool = False):
        """
        Args:
            use_simple_translator: SimpleTranslator 사용 (외부 의존성 최소화)
        """
        self.translator = create_translator(use_simple=use_simple_translator)
    
    def to_english(self, text: str) -> TranslationContext:
        """
        입력 텍스트를 영어로 번역합니다.
        
        Args:
            text: 원본 텍스트 (다국어)
            
        Returns:
            TranslationContext: 번역 컨텍스트 (원본 언어 정보 포함)
        """
        if not text or not text.strip():
            return TranslationContext(
                original_text=text,
                original_lang="en",
                english_text=text,
                is_translated=False
            )
        
        # 언어 감지
        detected_lang = detect_language(text)
        
        # 이미 영어면 번역 불필요
        if detected_lang == "en":
            return TranslationContext(
                original_text=text,
                original_lang="en",
                english_text=text,
                is_translated=False
            )
        
        # 영어로 번역
        try:
            english_text = self.translator.translate(
                text,
                src=detected_lang,
                dest="en"
            )
            logger.info(f"[Translation] {detected_lang} → en: {text[:50]}...")
            
            return TranslationContext(
                original_text=text,
                original_lang=detected_lang,
                english_text=english_text,
                is_translated=True
            )
        except Exception as e:
            logger.warning(f"영어 번역 실패: {e}")
            return TranslationContext(
                original_text=text,
                original_lang=detected_lang,
                english_text=text,  # 실패 시 원본 사용
                is_translated=False
            )
    
    def from_english(
        self,
        english_response: str,
        context: TranslationContext
    ) -> str:
        """
        영어 응답을 원래 언어로 번역합니다.
        
        Args:
            english_response: 모델의 영어 응답
            context: to_english()에서 반환된 컨텍스트
            
        Returns:
            원래 언어로 번역된 응답
        """
        # 원래 영어였으면 번역 불필요
        if not context.is_translated or context.original_lang == "en":
            return english_response
        
        if not english_response or not english_response.strip():
            return english_response
        
        # [CRITICAL FIX] 코드 블록(```)은 번역하지 않고 원문 유지
        # 파일명, 명령어 결과, stdout/stderr 등 기술적 데이터 보존을 위함
        import re
        
        # 코드 블록 패턴: ```...``` (멀티라인)
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, english_response)
        
        # 코드 블록을 플레이스홀더로 대체
        text_to_translate = english_response
        placeholders = []
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            placeholders.append((placeholder, block))
            text_to_translate = text_to_translate.replace(block, placeholder, 1)
        
        # 코드 블록 제외한 텍스트만 번역
        try:
            if text_to_translate.strip():
                translated = self.translator.translate(
                    text_to_translate,
                    src="en",
                    dest=context.original_lang
                )
            else:
                translated = text_to_translate
            
            # 코드 블록 복원 (원문 그대로)
            for placeholder, block in placeholders:
                translated = translated.replace(placeholder, block)
            
            logger.info(f"[Translation] en → {context.original_lang}: {english_response[:50]}... (preserved {len(code_blocks)} code blocks)")
            return translated
        except Exception as e:
            logger.warning(f"원래 언어 번역 실패: {e}")
            return english_response  # 실패 시 영어 응답 반환
    
    def process_with_model(
        self,
        text: str,
        model_fn: Callable[[str], str],
        force_translate_response: bool = True
    ) -> str:
        """
        번역 파이프라인을 통해 모델 처리
        
        Args:
            text: 사용자 입력 (다국어)
            model_fn: 모델 처리 함수 (영어 입력 → 영어 출력)
            force_translate_response: 응답도 번역할지 여부
            
        Returns:
            원래 언어로 된 응답
        """
        # 1. 영어로 변환
        ctx = self.to_english(text)
        
        # 2. 모델 처리
        english_response = model_fn(ctx.english_text)
        
        # 3. 원래 언어로 변환 (옵션)
        if force_translate_response:
            return self.from_english(english_response, ctx)
        else:
            return english_response


if __name__ == "__main__":
    # 테스트
    print("=== 번역 파이프라인 테스트 ===\n")
    
    pipeline = TranslationPipeline(use_simple_translator=True)
    
    # 한국어 입력 테스트
    korean_input = "안녕하세요, 오늘 서울 날씨가 어때요?"
    print(f"입력: {korean_input}")
    
    ctx = pipeline.to_english(korean_input)
    print(f"언어 감지: {ctx.original_lang}")
    print(f"영어 변환: {ctx.english_text}")
    print(f"번역됨: {ctx.is_translated}")
    
    # 모델 응답 시뮬레이션
    mock_response = "Today the weather in Seoul is cold, about -2 degrees Celsius with cloudy skies."
    print(f"\n모델 응답 (영어): {mock_response}")
    
    final = pipeline.from_english(mock_response, ctx)
    print(f"최종 응답 ({ctx.original_lang}): {final}")
