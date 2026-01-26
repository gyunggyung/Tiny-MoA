"""
Vector Store (ChromaDB Wrapper)
===============================
Lazy loading wrapper for ChromaDB to minimize startup overhead.
"""

import logging
from typing import Optional, Any
from pathlib import Path

class LazyVectorStore:
    """
    ChromaDB를 필요할 때만 로드하는 래퍼 클래스.
    MoA 시스템의 빠른 부팅 속도를 위해 필수적입니다.
    """
    
    def __init__(self, collection_name: str = "tiny_moa_docs"):
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_fn = None
        
    def _init_db(self):
        """DB 및 임베딩 모델 Lazy Initialization"""
        if self._client is not None:
            return
            
        print("[RAG] Initializing Vector Store (ChromaDB + SentenceTransformers)...")
        
        try:
            # [Compatibility Fix] numpy 2.0 removed np.float_ which chromadb < 0.6 uses
            import numpy as np
            if not hasattr(np, "float_"):
                np.float_ = np.float64
            
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Persistent Client (데이터 저장)
            db_path = Path("rag_storage")
            db_path.mkdir(exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            
            # Embedding Function (all-MiniLM-L6-v2: Small, Fast)
            # Default is all-MiniLM-L6-v2 which is great for CPU
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn
            )
            print("[RAG] Vector Store ready.")
            
        except ImportError as e:
            print(f"[RAG] Error loading dependencies: {e}")
            raise ImportError("RAG requires 'chromadb' and 'sentence-transformers'. Run `uv add chromadb sentence-transformers`.")

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """문서 추가"""
        self._init_db()
        if not documents:
            return
            
        print(f"[RAG] Adding {len(documents)} chunks to DB...")
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    def query(self, query_text: str, n_results: int = 3) -> dict:
        """유사 문서 검색"""
        self._init_db()
        
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def clear(self):
        """DB 초기화 (테스트용)"""
        self._init_db()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn
        )
