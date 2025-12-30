"""
Advanced hybrid retrieval system using ChromaDB
Combines vector similarity with BM25 keyword search and reranking
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# BM25 for keyword search
from rank_bm25 import BM25Okapi

from src.ingestion.chunking_strategy import Chunk
from src.utils.logger import setup_logger

# LLM imports for HYDE
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = setup_logger(__name__)


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

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document using HYDE (Hypothetical Document Embeddings)

        Args:
            query: Original user query

        Returns:
            Hypothetical document that would answer the query
        """
        if not self.config.use_hyde:
            return query

        try:
            # Format the prompt
            prompt = self.config.hyde_prompt_template.format(query=query)

            if self.config.llm_provider == 'openai' and OPENAI_AVAILABLE:
                # Use Azure OpenAI if endpoint is configured
                if hasattr(self.config, 'azure_openai_endpoint') and self.config.azure_openai_endpoint:
                    if not self.config.azure_openai_api_key:
                        logger.info("Azure OpenAI API key not available, using rule-based HYDE fallback")
                        return self._rule_based_hyde(query)
                    
                    client = AzureOpenAI(
                        api_key=self.config.azure_openai_api_key,
                        api_version="2025-01-01-preview",
                        azure_endpoint="https://intern-testing-resource.cognitiveservices.azure.com"
                    )
                    model_name = "intern-gpt-4o-mini"
                else:
                    if not self.config.openai_api_key or self.config.openai_api_key.startswith('dummy'):
                        logger.info("OpenAI API key not available, using rule-based HYDE fallback")
                        return self._rule_based_hyde(query)
                    
                    client = OpenAI(api_key=self.config.openai_api_key)
                    model_name = self.config.llm_model
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.llm_max_tokens,
                    temperature=self.config.llm_temperature
                )
                hypothetical_doc = response.choices[0].message.content.strip()

            elif self.config.llm_provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                if not self.config.anthropic_api_key:
                    logger.info("Anthropic API key not available, using rule-based HYDE fallback")
                    return self._rule_based_hyde(query)

                client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                response = client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=self.config.llm_max_tokens,
                    temperature=self.config.llm_temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                hypothetical_doc = response.content[0].text.strip()

            else:
                logger.info(f"LLM provider {self.config.llm_provider} not available, using rule-based HYDE fallback")
                return self._rule_based_hyde(query)

            logger.info(f"Generated hypothetical document ({len(hypothetical_doc)} chars)")
            return hypothetical_doc

        except Exception as e:
            logger.warning(f"HYDE generation failed: {e}, using rule-based fallback")
            return self._rule_based_hyde(query)

    def _rule_based_hyde(self, query: str) -> str:
        """
        Rule-based HYDE fallback that creates a hypothetical document
        based on common patterns in test case generation queries.
        """
        query_lower = query.lower()

        # Analyze query for key terms
        if any(term in query_lower for term in ['flight', 'booking', 'search', 'filter', 'airfare', 'travel']):
            hypothetical_doc = f"""
# Flight Booking Search and Filtering System

## Functional Requirements

### Flight Search Process
The system must provide comprehensive flight search capabilities with advanced filtering options. Users can search for flights based on origin, destination, travel dates, passenger count, and cabin class.

### Search Criteria
- Origin and destination airports (IATA codes supported)
- Departure and return dates
- Number of passengers (adults, children, infants)
- Cabin class (Economy, Premium Economy, Business, First Class)
- Preferred airlines or alliance partnerships

### Filter Options
- Price range filtering with minimum and maximum fare limits
- Number of stops (Non-stop, 1 stop, 2+ stops)
- Airline selection (multiple airlines can be selected)
- Departure/arrival time ranges (Morning, Afternoon, Evening, Night)
- Baggage inclusion (carry-on, checked baggage)
- Refundable fares option
- Seat availability and preferred seat types

### Dynamic Results Update
When filters are applied, flight results must update dynamically without requiring a new search. The system shall maintain search context while applying progressive filters.

### Filter Reset Functionality
Users must be able to clear all applied filters with a single action, returning to the original search results. Individual filters can also be removed selectively.

### Search Result Sorting
Results can be sorted by:
- Price (lowest first)
- Duration (shortest first)
- Departure time (earliest first)
- Arrival time (earliest first)
- Airline preference
- Number of stops (fewest first)

### Real-time Price Updates
Flight prices may change dynamically. The system must handle price updates gracefully, notifying users of fare changes during the booking process.

### Error Handling
- No flights found: Clear messaging with suggestions to modify search criteria
- Invalid search criteria: Validation messages for impossible date ranges or invalid airport codes
- Network connectivity issues: Graceful degradation with cached results when possible
- High demand periods: Queue management and estimated wait times

### Performance Requirements
- Search response time under 3 seconds for typical queries
- Filter application under 1 second
- Support for concurrent users during peak booking periods
- Mobile-responsive design for all filtering operations

### Accessibility
- WCAG 2.1 AA compliance for all filter interfaces
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
"""

        else:
            # Generic fallback for other queries
            hypothetical_doc = f"""
# Software Requirements Document

## Overview
This document outlines the requirements for implementing {query} functionality in the system.

## Functional Requirements

### Core Features
- The system must provide {query} capabilities
- All operations must be performed securely
- User interface must be intuitive and responsive
- Error handling must be comprehensive

### User Experience
- Clear feedback for all user actions
- Loading indicators for asynchronous operations
- Consistent design language throughout
- Accessibility compliance (WCAG 2.1 AA)

### Data Management
- All data must be validated before processing
- Secure storage with appropriate encryption
- Audit logging for all critical operations
- Data backup and recovery procedures

## Technical Requirements

### Performance
- Response times under 2 seconds for typical operations
- Support for concurrent users
- Efficient database queries and indexing
- Caching strategies for improved performance

### Security
- Input validation and sanitization
- Authentication and authorization checks
- HTTPS encryption for all communications
- Regular security updates and patches

### Scalability
- Horizontal scaling capabilities
- Database optimization and indexing
- CDN integration for static assets
- Microservices architecture consideration
"""

        logger.info(f"Generated rule-based hypothetical document ({len(hypothetical_doc)} chars)")
        return hypothetical_doc

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
                use_hybrid: bool = True,
                use_hyde: bool = None) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
            use_hybrid: Whether to use hybrid retrieval
            use_hyde: Whether to use HYDE (Hypothetical Document Embeddings).
                     If None, uses config.use_hyde setting.

        Returns:
            List of relevant Chunk objects with scores
        """
        if not self.chunk_map:
            logger.warning("No chunks indexed")
            return []

        # Determine whether to use HYDE
        hyde_enabled = use_hyde if use_hyde is not None else self.config.use_hyde

        # Generate hypothetical document if HYDE is enabled
        retrieval_query = query
        if hyde_enabled:
            logger.info(f"Generating hypothetical document for query: {query[:100]}")
            retrieval_query = self._generate_hypothetical_document(query)
            logger.info(f"Using hypothetical document ({len(retrieval_query)} chars) for retrieval")

        logger.info(f"Retrieving top {top_k} chunks{' with HYDE' if hyde_enabled else ''}")

        if use_hybrid and self.bm25_index:
            # Hybrid retrieval (vector + BM25 + RRF)
            results = self._hybrid_retrieve(retrieval_query, top_k, min_score)
        else:
            # Vector-only retrieval
            results = self._vector_retrieve(retrieval_query, top_k, min_score)
        
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
