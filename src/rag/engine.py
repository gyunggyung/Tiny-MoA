"""
RAG Engine
==========
Orchestrates Document Processing (Docling) -> Embedding (Chroma) -> Retrieval
"""

import os
from pathlib import Path
from typing import List

# Internal imports (Lazy loaded components)
from src.rag.store import LazyVectorStore
from src.doc_processing.converter import DoclingConverter

import hashlib

class RAGEngine:
    def __init__(self):
        self.store = LazyVectorStore()
        # Docling Converter needs to be instantiated carefully (it handles its own lazy imports)
        self.converter = None # Lazy init

    def _get_converter(self):
        if self.converter is None:
            self.converter = DoclingConverter(high_speed=True)
        return self.converter

    def ingest_file(self, file_path: str) -> str:
        """
        파일을 읽어서 처리하고 벡터 DB에 저장
        Returns:
            summary or status message
        """
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found {file_path}"
            
        print(f"[RAG] Processing file: {path.name}")
        
        # 1. Convert to Markdown
        converter = self._get_converter()
        try:
            full_text = converter.convert(str(path))
        except Exception as e:
            return f"Error converting file: {e}"
            
        # 2. Chunking (Simple paragraph/header based split for now)
        # Markdown splits nicely by newlines usually.
        # Simple Logic: Split by double newlines, group into ~500 char chunks
        chunks = self._simple_chunker(full_text)
        
        # 3. Create IDs and Metadata
        file_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]
        
        # 4. Store
        try:
            self.store.add_documents(chunks, metadatas, ids)
            return f"Successfully ingested {path.name} ({len(chunks)} chunks)."
        except Exception as e:
            return f"Error storing to DB: {e}"

    def query(self, query_text: str) -> str:
        """
        질문과 관련된 문서 검색
        Returns:
            Combine context string
        """
        try:
            results = self.store.query(query_text, n_results=3)
        except ImportError:
             return "RAG System not initialized (dependencies missing)."
        except Exception as e:
             return f"Error querying DB: {e}"
        
        if not results or not results['documents']:
            return ""
            
        # Flatten results
        docs = results['documents'][0] # list of list
        context = "\n\n---\n".join(docs)
        return context

    def _simple_chunker(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """간단한 청킹 로직"""
        # 1. Split by paragraphs (double newline)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            
            # If a single paragraph is huge, split it blindly (fallback)
            if len(para) > chunk_size:
                # Add existing buffer first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(para) # TODO: Better splitting for huge paragraphs
                continue
                
            if current_len + len(para) > chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para)
                
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks
