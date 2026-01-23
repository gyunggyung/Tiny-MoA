"""
Google Translator
==================
Google Translate API 래퍼 (무료 버전 - googletrans)
"""

import logging
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class GoogleTranslator:
    """
    Google Translate를 사용한 번역기
    
    googletrans 패키지 사용 (무료, API 키 불필요)
    주의: 대량 요청 시 rate limit 가능
    """
    
    def __init__(self):
        self._translator = None
    
    @property
    def translator(self):
        """Lazy initialization of translator"""
        if self._translator is None:
            try:
                from googletrans import Translator
                self._translator = Translator()
            except ImportError:
                raise ImportError(
                    "googletrans 패키지가 필요합니다:\n"
                    "pip install googletrans==4.0.0-rc1"
                )
        return self._translator
    
    def translate(
        self,
        text: str,
        src: str = "auto",
        dest: str = "en",
    ) -> str:
        """
        단일 텍스트 번역
        
        Args:
            text: 번역할 텍스트
            src: 원본 언어 코드 ('auto'면 자동 감지)
            dest: 대상 언어 코드
            
        Returns:
            번역된 텍스트
        """
        if not text or not text.strip():
            return text
        
        try:
            result = self.translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            logger.warning(f"번역 실패: {e}")
            return text  # 실패 시 원본 반환
    
    def translate_batch(
        self,
        texts: List[str],
        src: str = "auto",
        dest: str = "en",
        max_workers: int = 1,
    ) -> List[str]:
        """
        여러 텍스트 일괄 번역
        
        Args:
            texts: 번역할 텍스트 리스트
            src: 원본 언어 코드
            dest: 대상 언어 코드
            max_workers: 병렬 처리 워커 수 (주의: rate limit)
            
        Returns:
            번역된 텍스트 리스트
        """
        if not texts:
            return []
        
        # 순차 처리 (rate limit 방지)
        if max_workers <= 1:
            return [self.translate(t, src, dest) for t in texts]
        
        # 병렬 처리 (주의: rate limit 가능성)
        results = [""] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.translate, text, src, dest): i
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"번역 실패 (idx={idx}): {e}")
                    results[idx] = texts[idx]
        
        return results
    
    def detect(self, text: str) -> str:
        """
        언어 감지 (Google Translate API 사용)
        
        Args:
            text: 감지할 텍스트
            
        Returns:
            언어 코드
        """
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            logger.warning(f"언어 감지 실패: {e}")
            return "en"


class SimpleTranslator:
    """
    간단한 번역기 (외부 의존성 없음)
    
    requests만 사용하여 Google Translate 웹 API 호출
    """
    
    def translate(
        self,
        text: str,
        src: str = "auto",
        dest: str = "en",
    ) -> str:
        """
        단일 텍스트 번역 (requests 사용)
        """
        import requests
        
        if not text or not text.strip():
            return text
        
        try:
            # Google Translate 웹 API (비공식)
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": src,
                "tl": dest,
                "dt": "t",
                "q": text
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # 응답 파싱 (중첩 리스트 형태)
            result = response.json()
            translated = "".join(
                part[0] for part in result[0] if part[0]
            )
            return translated
            
        except Exception as e:
            logger.warning(f"SimpleTranslator 번역 실패: {e}")
            return text


def create_translator(use_simple: bool = False):
    """
    번역기 팩토리 함수
    
    Args:
        use_simple: SimpleTranslator 사용 여부
        
    Returns:
        번역기 인스턴스
    """
    if use_simple:
        return SimpleTranslator()
    
    try:
        return GoogleTranslator()
    except ImportError:
        logger.warning("googletrans 없음, SimpleTranslator 사용")
        return SimpleTranslator()


if __name__ == "__main__":
    # 테스트
    print("=== 번역기 테스트 ===\n")
    
    # SimpleTranslator 테스트 (외부 의존성 없음)
    simple = SimpleTranslator()
    
    test_cases = [
        ("Hello, how are you?", "en", "ko"),
        ("안녕하세요", "ko", "en"),
        ("오늘 날씨가 좋네요", "ko", "en"),
    ]
    
    for text, src, dest in test_cases:
        result = simple.translate(text, src, dest)
        print(f"[{src}→{dest}] {text} → {result}")
