"""
Advanced chunking strategies for document processing
Implements semantic and hybrid chunking with overlap
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
import hashlib

from src.ingestion.document_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default="")
    
    def __post_init__(self):
        """Generate chunk_id if not provided"""
        if not self.chunk_id:
            # Generate hash from content
            hash_input = f"{self.metadata.get('doc_id', '')}:{self.content[:500]}"
            self.chunk_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class HybridChunkingStrategy:
    """
    Advanced hybrid chunking strategy that combines:
    - Semantic boundaries (paragraphs, sections)
    - Fixed-size chunking with overlap
    - Header-aware chunking
    """
    
    def __init__(self, config):
        """Initialize chunking strategy"""
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.min_chunk_size = config.min_chunk_size
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a document using hybrid strategy
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        content = document.content
        
        if not content or len(content) < self.min_chunk_size:
            logger.warning(f"Document too small to chunk: {len(content)} chars")
            return []
        
        # Try semantic chunking first
        chunks = self._semantic_chunk(content, document.metadata)
        
        # If semantic chunking produces large chunks, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.chunk_size * 1.5:
                # Split large chunks with overlap
                sub_chunks = self._fixed_size_chunk(
                    chunk.content,
                    chunk.metadata
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # Deduplicate similar chunks
        final_chunks = self._deduplicate_chunks(final_chunks)
        
        logger.info(f"Created {len(final_chunks)} chunks from document {document.doc_id}")
        
        return final_chunks
    
    def _semantic_chunk(self, content: str, base_metadata: Dict) -> List[Chunk]:
        """Chunk by semantic boundaries (paragraphs, sections)"""
        chunks = []
        
        # Split by double newlines (paragraphs)
        sections = re.split(r'\n\n+', content)
        
        current_chunk = ""
        current_header = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Detect headers (lines that look like titles)
            if self._is_header(section):
                # Start new chunk with this header
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **base_metadata,
                            'header': current_header,
                            'chunk_type': 'semantic'
                        }
                    )
                    chunks.append(chunk)
                
                current_header = section
                current_chunk = section + "\n\n"
            else:
                # Add to current chunk
                current_chunk += section + "\n\n"
                
                # If chunk is getting large, split it
                if len(current_chunk) > self.chunk_size:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **base_metadata,
                            'header': current_header,
                            'chunk_type': 'semantic'
                        }
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with context (overlap)
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                metadata={
                    **base_metadata,
                    'header': current_header,
                    'chunk_type': 'semantic'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunk(self, content: str, base_metadata: Dict) -> List[Chunk]:
        """Chunk by fixed size with overlap"""
        chunks = []
        
        # Split into sentences for cleaner boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **base_metadata,
                            'chunk_type': 'fixed_size'
                        }
                    )
                    chunks.append(chunk)
                    
                    # Create overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + sentence + " "
                else:
                    # Sentence itself is too long, split it
                    if len(sentence) > self.chunk_size:
                        words = sentence.split()
                        for i in range(0, len(words), self.chunk_size // 10):
                            chunk_words = words[i:i + self.chunk_size // 10]
                            chunk_text = " ".join(chunk_words)
                            chunk = Chunk(
                                content=chunk_text,
                                metadata={
                                    **base_metadata,
                                    'chunk_type': 'fixed_size_word'
                                }
                            )
                            chunks.append(chunk)
                    else:
                        current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                metadata={
                    **base_metadata,
                    'chunk_type': 'fixed_size'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to get overlap at sentence boundary
        overlap_text = text[-self.chunk_overlap:]
        
        # Find last sentence start
        last_period = overlap_text.rfind('. ')
        if last_period > self.chunk_overlap // 2:
            overlap_text = overlap_text[last_period + 2:]
        
        return overlap_text
    
    def _is_header(self, text: str) -> bool:
        """Detect if text is likely a header"""
        # Headers are typically:
        # - Short (< 100 chars)
        # - Single line or very few lines
        # - May start with #, numbers, or be ALL CAPS
        # - Don't end with punctuation
        
        if len(text) > 100:
            return False
        
        lines = text.split('\n')
        if len(lines) > 3:
            return False
        
        first_line = lines[0].strip()
        
        # Check for markdown headers
        if first_line.startswith('#'):
            return True
        
        # Check for numbered headers
        if re.match(r'^\d+\.?\s+[A-Z]', first_line):
            return True
        
        # Check for all caps (likely header)
        if first_line.isupper() and len(first_line.split()) < 10:
            return True
        
        # Check if ends without punctuation and is short
        if not first_line.endswith(('.', '!', '?', ',')) and len(first_line.split()) < 12:
            # Check if starts with capital letter
            if first_line and first_line[0].isupper():
                return True
        
        return False
    
    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove near-duplicate chunks"""
        if not chunks:
            return chunks
        
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Create a normalized version for comparison
            normalized = self._normalize_for_comparison(chunk.content)
            
            # Check if we've seen very similar content
            is_duplicate = False
            for seen in seen_content:
                similarity = self._content_similarity(normalized, seen)
                if similarity > 0.85:  # 85% similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_content.add(normalized)
        
        removed = len(chunks) - len(unique_chunks)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate chunks")
        
        return unique_chunks
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for similarity comparison"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple Jaccard similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
