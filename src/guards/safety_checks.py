"""
Safety guards and checks for RAG system
Includes hallucination detection, prompt injection prevention, and evidence quality checks
"""

import re
from typing import List, Dict, Tuple, Any

from src.ingestion.chunking_strategy import Chunk
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafetyGuard:
    """Safety checks and guardrails for RAG system"""
    
    def __init__(self, config):
        """Initialize safety guard"""
        self.config = config
        
        # Prompt injection patterns to detect
        self.injection_patterns = [
            r'ignore\s+(previous|all|above|prior)\s+instructions?',
            r'forget\s+(everything|all|what)',
            r'you\s+are\s+now',
            r'new\s+instructions?',
            r'disregard\s+(previous|all|above)',
            r'system\s*:\s*',
            r'<\s*admin\s*>',
            r'sudo\s+mode',
            r'developer\s+mode',
            r'jailbreak',
        ]
        
        # Compile patterns
        self.injection_regex = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.injection_patterns
        ]
    
    def check_query_safety(self, query: str) -> Tuple[bool, str]:
        """
        Check if query is safe (no prompt injection attempts)
        
        Args:
            query: User query
            
        Returns:
            Tuple of (is_safe, reason)
        """
        query_lower = query.lower()
        
        # Check for prompt injection patterns
        for pattern in self.injection_regex:
            if pattern.search(query):
                logger.warning(f"Potential prompt injection detected: {pattern.pattern}")
                return False, "Query contains suspicious patterns and cannot be processed"
        
        # Check for extremely long queries (potential attack)
        if len(query) > 10000:
            logger.warning(f"Query too long: {len(query)} characters")
            return False, "Query is too long"
        
        # Check for excessive special characters (potential encoding attack)
        special_char_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace()) / max(len(query), 1)
        if special_char_ratio > 0.3:
            logger.warning(f"Too many special characters: {special_char_ratio:.2%}")
            return False, "Query contains too many special characters"
        
        return True, "Query is safe"
    
    def check_evidence_quality(self, chunks: List[Chunk], min_chunks: int = 2) -> Tuple[bool, str]:
        """
        Check if retrieved evidence is sufficient
        
        Args:
            chunks: Retrieved chunks
            min_chunks: Minimum number of chunks required
            
        Returns:
            Tuple of (is_sufficient, reason)
        """
        if not chunks:
            return False, "No relevant context found. Please provide more information or upload relevant documents."
        
        if len(chunks) < min_chunks:
            return False, f"Found only {len(chunks)} relevant chunks. Need at least {min_chunks}. Please provide more context."
        
        # Check average relevance score
        avg_score = sum(c.metadata.get('score', 0) for c in chunks) / len(chunks)
        
        if avg_score < 0.3:
            return False, f"Retrieved context has low relevance (avg score: {avg_score:.3f}). Please rephrase your query or provide more specific information."
        
        # Check for minimum content length
        total_length = sum(len(c.content) for c in chunks)
        if total_length < 200:
            return False, "Retrieved context is too short. Please provide more detailed documentation."
        
        return True, "Evidence quality is sufficient"
    
    def check_hallucination(self, use_cases: List[Dict], chunks: List[Chunk]) -> Tuple[bool, str]:
        """
        Check if generated use cases are grounded in retrieved context
        
        Args:
            use_cases: Generated use cases
            chunks: Retrieved context chunks
            
        Returns:
            Tuple of (is_grounded, reason)
        """
        if not self.config.safety_check_hallucination:
            return True, "Hallucination check disabled"
        
        if not use_cases:
            return True, "No use cases to check"
        
        # Extract all text from chunks
        context_text = " ".join(c.content.lower() for c in chunks)
        context_words = set(context_text.split())
        
        # Check each use case
        hallucination_warnings = []
        
        for uc in use_cases:
            # Extract key terms from use case
            uc_text = self._extract_use_case_text(uc).lower()
            uc_words = set(uc_text.split())
            
            # Check overlap with context
            common_words = uc_words.intersection(context_words)
            
            # Remove common stop words for more accurate check
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                         'user', 'system', 'should', 'must', 'can', 'will', 'click', 'enter',
                         'open', 'page', 'test', 'case', 'step', 'expected', 'result'}
            
            uc_words_filtered = uc_words - stop_words
            common_words_filtered = common_words - stop_words
            
            # Calculate overlap ratio
            if len(uc_words_filtered) > 0:
                overlap_ratio = len(common_words_filtered) / len(uc_words_filtered)
                
                if overlap_ratio < 0.3:
                    warning = f"Use case '{uc.get('title', 'unknown')}' may contain hallucinated content (low context overlap: {overlap_ratio:.2%})"
                    hallucination_warnings.append(warning)
                    logger.warning(warning)
        
        if hallucination_warnings:
            return False, "; ".join(hallucination_warnings)
        
        return True, "All use cases appear grounded in context"
    
    def _extract_use_case_text(self, use_case: Dict) -> str:
        """Extract all text from a use case"""
        text_parts = []
        
        # Add title and goal
        text_parts.append(use_case.get('title', ''))
        text_parts.append(use_case.get('goal', ''))
        
        # Add steps
        if 'steps' in use_case and isinstance(use_case['steps'], list):
            text_parts.extend(use_case['steps'])
        
        # Add expected results
        if 'expected_results' in use_case and isinstance(use_case['expected_results'], list):
            text_parts.extend(use_case['expected_results'])
        
        # Add preconditions
        if 'preconditions' in use_case and isinstance(use_case['preconditions'], list):
            text_parts.extend(use_case['preconditions'])
        
        return " ".join(str(p) for p in text_parts)
    
    def check_document_content(self, content: str) -> Tuple[bool, str]:
        """
        Check document content for prompt injection attempts
        
        Args:
            content: Document content
            
        Returns:
            Tuple of (is_safe, reason)
        """
        content_lower = content.lower()
        
        # Check for prompt injection in documents
        for pattern in self.injection_regex:
            matches = pattern.findall(content)
            if len(matches) > 2:  # Allow some false positives
                logger.warning(f"Document contains suspicious patterns: {pattern.pattern}")
                return False, f"Document contains suspicious instructions: {pattern.pattern}"
        
        return True, "Document content is safe"
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to prevent injection
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Sanitized metadata
        """
        sanitized = {}
        
        for key, value in metadata.items():
            # Only keep safe types
            if isinstance(value, (str, int, float, bool)):
                # Sanitize strings
                if isinstance(value, str):
                    # Remove control characters
                    value = ''.join(char for char in value if char.isprintable() or char in '\n\t')
                    # Limit length
                    value = value[:1000]
                
                sanitized[key] = value
        
        return sanitized
