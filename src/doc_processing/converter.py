"""
Docling Converter
=================
PDF/DOCX 문서를 텍스트로 변환하여 Brain에게 전달
"""

import logging
import os
from pathlib import Path
from typing import Optional

# Docling imports
try:
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        WordFormatOption,
        PowerpointFormatOption,
        HTMLFormatOption,
        ImageFormatOption
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    # Define dummy for type hints if needed, or just let it fail at runtime
    if not DOCLING_AVAILABLE:
         logging.warning("Docling not installed. DoclingConverter will fail.")
         DocumentConverter = None  # type: ignore


class DoclingConverter:
    """Docling 기반 문서 변환기"""
    
    def __init__(self, high_speed: bool = True):
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not installed. Run `uv add docling`.")
            
        self.high_speed = high_speed
        self._converter = self._create_converter()
        
    def _create_converter(self) -> DocumentConverter:
        """변환기 초기화 (Fast Mode 최적화)"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False # OCR 끄기 (속도 향상)
        pipeline_options.do_table_structure = True # 표 구조 분석
        
        if self.high_speed:
            # Fast Mode: 이미지 생성 끄고 저해상도 처리
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
            pipeline_options.generate_picture_images = False
            pipeline_options.generate_table_images = False
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_table_images = True

        # Fast Mode일 경우 pypdfium2 백엔드 사용 (3-5배 가속)
        if self.high_speed:
            try:
                pdf_format_option = PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend 
                )
            except ImportError:
                print("Warning: PyPdfiumDocumentBackend not found using default.")
                pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
        else:
            pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.HTML],
            format_options={
                InputFormat.PDF: pdf_format_option,
                InputFormat.DOCX: WordFormatOption(),
                InputFormat.PPTX: PowerpointFormatOption(),
                InputFormat.HTML: HTMLFormatOption(),
            },
        )

    def convert(self, file_path: str) -> str:
        """
        문서를 변환하여 마크다운 텍스트 반환
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if path.suffix.lower() in [".md", ".txt"]:
            self.logger.info(f"[{self.name}] Native {path.suffix} detected. Skipping Docling engine.")
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
                
        print(f"[Docling] Converting {path.name} (FastMode={self.high_speed})...")
        
        # 1. 변환 실행
        result = self._converter.convert(path)
        
        # 2. Markdown 추출
        md_text = result.document.export_to_markdown()
        
        # 3. (Optional) 후처리
        # 너무 긴 문서는 Brain 컨텍스트에 맞게 자르거나 요약해야 하지만, 
        # 일단은 원본 전체 반환 (Brain이 알아서 일부만 보거나 RAG로 넘겨야 함)
        
        return md_text

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        converter = DoclingConverter(high_speed=True)
        print(converter.convert(sys.argv[1])[:500] + "...")
