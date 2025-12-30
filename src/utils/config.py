"""
Configuration and utility classes for the RAG system
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import yaml

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# BM25 for keyword search
from rank_bm25 import BM25Okapi

from src.ingestion.chunking_strategy import Chunk
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Config:
    """Configuration class for the RAG system"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        self.config_path = config_path

        # Default configuration
        self.default_config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'chroma_persist_directory': './data/chroma_db',
            'chroma_collection_name': 'rag_documents',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'min_chunk_size': 100,
            'max_chunk_size': 1000,
            'top_k_vector': 20,
            'top_k_bm25': 20,
            'final_top_k': 5,
            'min_score_threshold': 0.3,
            'retrieval_use_reranking': True,
            'rerank': True,
            'rerank_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'llm_provider': 'openai',
            'llm_model': 'gpt-3.5-turbo',
            'llm_max_tokens': 2000,
            'llm_temperature': 0.7,
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            # Azure OpenAI configuration
            'azure_openai_endpoint': 'https://intern-testing-resource.cognitiveservices.azure.com/openai/deployments/intern-gpt-4o-mini/chat/completions?api-version=2025-01-01-preview',
            'azure_openai_api_key': os.getenv('AZURE_OPENAI_API_KEY', ''),
            'azure_deployment_name': 'intern-gpt-4o-mini',
            # HYDE (Hypothetical Document Embeddings) settings
            'use_hyde': True,
            'hyde_prompt_template': '''Given the query: "{query}"

Generate a hypothetical document that would be highly relevant to answer this query. The document should contain detailed, comprehensive information about the topic, including specific examples, requirements, and implementation details that would naturally appear in documentation about this subject.

Focus on creating content that matches the style and depth of technical documentation, API specifications, requirements documents, or test cases related to software development and testing.

Hypothetical document:''',
            # OCR settings for image processing
            'ocr_enabled': True,
            'ocr_languages': ['eng'],
            # Safety settings
            'safety_check_hallucination': True,
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY', ''),
                'anthropic': os.getenv('ANTHROPIC_API_KEY', '')
            }
        }


class HybridRetriever:
    """
    Hybrid retrieval system combining:
    1. Dense vector search (ChromaDB with sentence transformers)
    2. Sparse keyword search (BM25)
    3. Reciprocal Rank Fusion (RRF) for combining results
    4. Optional reranking
    """
    
    def __init__(self, config):
        """Initialize hybrid retriever"""
        self.config = config
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
        # Initialize ChromaDB
        persist_dir = Path(config.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {persist_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index (will be built when chunks are indexed)
        self.bm25_index = None
        self.bm25_chunks = []
        self.chunk_map = {}  # Maps chunk_id to Chunk object
        
        # Load existing chunks if any
        self._load_existing_chunks()
        
        logger.info(f"Retriever initialized with {len(self.chunk_map)} chunks")
    
    def _load_existing_chunks(self):
        """Load existing chunks from ChromaDB"""
        try:
            # Get all documents from collection
            results = self.collection.get(include=['documents', 'metadatas'])
            
            if results['ids']:
                logger.info(f"Loading {len(results['ids'])} existing chunks")
                
                # Reconstruct chunks
                for chunk_id, content, metadata in zip(
                    results['ids'],
                    results['documents'],
                    results['metadatas']
                ):
                    chunk = Chunk(
                        content=content,
                        metadata=metadata or {},
                        chunk_id=chunk_id
                    )
                    self.chunk_map[chunk_id] = chunk
                    self.bm25_chunks.append(chunk)
                
                # Rebuild BM25 index
                if self.bm25_chunks:
                    self._build_bm25_index(self.bm25_chunks)
        
        except Exception as e:
            logger.warning(f"Could not load existing chunks: {e}")
    
    def index_chunks(self, chunks: List[Chunk], force_reindex: bool = False):
        """
        Index chunks for retrieval
        
        Args:
            chunks: List of Chunk objects to index
            force_reindex: Whether to clear existing index
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        if force_reindex:
            logger.info("Force reindex: clearing existing data")
            self.collection.delete(where={})
            self.chunk_map.clear()
            self.bm25_chunks.clear()
        
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        # Filter out already indexed chunks
        new_chunks = [c for c in chunks if c.chunk_id not in self.chunk_map]
        
        if not new_chunks:
            logger.info("All chunks already indexed")
            return
        
        logger.info(f"Indexing {len(new_chunks)} new chunks")
        
        # Generate embeddings in batches
        batch_size = 32
        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []
        chunk_embeddings = []
        
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_texts = [c.content for c in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            for chunk, embedding in zip(batch, embeddings):
                chunk_ids.append(chunk.chunk_id)
                chunk_contents.append(chunk.content)
                chunk_metadatas.append(chunk.metadata)
                chunk_embeddings.append(embedding.tolist())
                
                # Store in memory
                self.chunk_map[chunk.chunk_id] = chunk
                self.bm25_chunks.append(chunk)
        
        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_contents,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings
        )
        
        # Rebuild BM25 index
        self._build_bm25_index(self.bm25_chunks)
        
        logger.info(f"Successfully indexed {len(new_chunks)} chunks")
        logger.info(f"Total chunks in index: {len(self.chunk_map)}")
    
    def _build_bm25_index(self, chunks: List[Chunk]):
        """Build BM25 index for keyword search"""
        logger.info("Building BM25 index...")
        
        # Tokenize chunks
        tokenized_chunks = [chunk.content.lower().split() for chunk in chunks]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
        logger.info("BM25 index built")
    
    def retrieve(self,
                query: str,
                top_k: int = 5,
                min_score: float = 0.3,
                use_hybrid: bool = True) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
            use_hybrid: Whether to use hybrid retrieval
            
        Returns:
            List of relevant Chunk objects with scores
        """
        if not self.chunk_map:
            logger.warning("No chunks indexed")
            return []
        
        logger.info(f"Retrieving top {top_k} chunks for query: {query[:100]}")
        
        if use_hybrid and self.bm25_index:
            # Hybrid retrieval (vector + BM25 + RRF)
            results = self._hybrid_retrieve(query, top_k, min_score)
        else:
            # Vector-only retrieval
            results = self._vector_retrieve(query, top_k, min_score)
        
        # Optionally rerank
        if self.config.retrieval_use_reranking and len(results) > 1:
            results = self._rerank(query, results)
        
        logger.info(f"Retrieved {len(results)} chunks")
        
        return results
    
    def _vector_retrieve(self, query: str, top_k: int, min_score: float) -> List[Chunk]:
        """Retrieve using vector similarity only"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, len(self.chunk_map)),  # Get more for filtering
            include=['distances', 'metadatas']
        )
        
        # Convert distances to similarity scores (cosine distance -> similarity)
        chunks = []
        for chunk_id, distance in zip(results['ids'][0], results['distances'][0]):
            similarity = 1 - distance  # Convert distance to similarity
            
            if similarity >= min_score:
                chunk = self.chunk_map.get(chunk_id)
                if chunk:
                    # Add score to metadata
                    chunk.metadata['score'] = similarity
                    chunk.metadata['retrieval_method'] = 'vector'
                    chunks.append(chunk)
        
        # Sort by score and limit
        chunks.sort(key=lambda c: c.metadata.get('score', 0), reverse=True)
        
        return chunks[:top_k]
    
    def _hybrid_retrieve(self, query: str, top_k: int, min_score: float) -> List[Chunk]:
        """Retrieve using hybrid approach (vector + BM25 + RRF)"""
        k = top_k * 3  # Get more candidates for fusion
        
        # 1. Vector retrieval
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, len(self.chunk_map)),
            include=['distances']
        )
        
        vector_ranks = {}
        for rank, (chunk_id, distance) in enumerate(
            zip(vector_results['ids'][0], vector_results['distances'][0]), 1
        ):
            similarity = 1 - distance
            if similarity >= min_score:
                vector_ranks[chunk_id] = (rank, similarity)
        
        # 2. BM25 retrieval
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
        
        bm25_ranks = {}
        for rank, idx in enumerate(bm25_top_indices, 1):
            if idx < len(self.bm25_chunks):
                chunk = self.bm25_chunks[idx]
                bm25_score = bm25_scores[idx]
                bm25_ranks[chunk.chunk_id] = (rank, bm25_score)
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_rrf = 60  # RRF constant
        
        # Combine vector ranks
        for chunk_id, (rank, score) in vector_ranks.items():
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 / (k_rrf + rank))
        
        # Combine BM25 ranks
        for chunk_id, (rank, score) in bm25_ranks.items():
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 / (k_rrf + rank))
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top chunks
        chunks = []
        for chunk_id, rrf_score in sorted_chunk_ids[:top_k]:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                # Add combined score
                vector_score = vector_ranks.get(chunk_id, (999, 0))[1]
                bm25_score = bm25_ranks.get(chunk_id, (999, 0))[1]
                
                chunk.metadata['score'] = rrf_score
                chunk.metadata['vector_score'] = vector_score
                chunk.metadata['bm25_score'] = bm25_score
                chunk.metadata['retrieval_method'] = 'hybrid'
                
                chunks.append(chunk)
        
        return chunks
    
    def _rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank chunks using cross-encoder or similarity-based reranking
        Simple implementation using query-chunk similarity
        """
        if not chunks:
            return chunks
        
        logger.info("Reranking results...")
        
        # Calculate query-specific scores
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        for chunk in chunks:
            chunk_embedding = self.embedding_model.encode(
                chunk.content,
                convert_to_numpy=True
            )
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            
            # Combine with existing score
            original_score = chunk.metadata.get('score', 0)
            chunk.metadata['score'] = 0.7 * similarity + 0.3 * original_score
            chunk.metadata['reranked'] = True
        
        # Sort by new score
        chunks.sort(key=lambda c: c.metadata.get('score', 0), reverse=True)
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            'total_chunks': len(self.chunk_map),
            'collection_name': self.config.chroma_collection_name,
            'embedding_model': self.config.embedding_model,
            'bm25_enabled': self.bm25_index is not None
        }
