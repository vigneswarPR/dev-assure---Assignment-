"""
Multimodal RAG System for Test Case Generation
Main application entry point with CLI and API interfaces
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.chunking_strategy import HybridChunkingStrategy
from src.retrieval.retriever import HybridRetriever
from src.generation.use_case_generator import UseCaseGenerator
from src.guards.safety_checks import SafetyGuard
from src.utils.logger import setup_logger
from src.utils.config import Config

# Setup logging
logger = setup_logger(__name__)


class RAGApplication:
    """Main RAG application orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the RAG application"""
        self.config = Config(config_path)
        self.doc_processor = DocumentProcessor(self.config)
        self.chunking_strategy = HybridChunkingStrategy(self.config)
        self.retriever = HybridRetriever(self.config)
        self.generator = UseCaseGenerator(self.config)
        self.safety_guard = SafetyGuard(self.config)
        
        logger.info("RAG Application initialized successfully")
    
    def ingest_documents(self, folder_path: str, force_reindex: bool = False) -> Dict:
        """
        Ingest documents from a folder
        
        Args:
            folder_path: Path to folder containing documents
            force_reindex: Whether to force reindexing of all documents
            
        Returns:
            Dictionary with ingestion statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting document ingestion from: {folder_path}")
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Get all supported files
        supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.png', '.jpg', '.jpeg'}
        files = [f for f in folder.rglob('*') if f.suffix.lower() in supported_extensions]
        
        logger.info(f"Found {len(files)} files to process")
        
        stats = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        all_chunks = []
        
        for file_path in files:
            try:
                logger.info(f"Processing: {file_path.name}")
                
                # Process document
                documents = self.doc_processor.process_file(str(file_path))
                
                # Chunk documents
                for doc in documents:
                    chunks = self.chunking_strategy.chunk_document(doc)
                    all_chunks.extend(chunks)
                
                stats['processed'] += 1
                logger.info(f"Processed {file_path.name} - {len(chunks)} chunks")
                
            except Exception as e:
                stats['failed'] += 1
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                stats['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Index all chunks
        if all_chunks:
            logger.info(f"Indexing {len(all_chunks)} chunks...")
            self.retriever.index_chunks(all_chunks, force_reindex=force_reindex)
            stats['total_chunks'] = len(all_chunks)
        
        duration = (datetime.now() - start_time).total_seconds()
        stats['duration_seconds'] = duration
        
        logger.info(f"Ingestion complete: {stats['processed']}/{stats['total_files']} files, "
                   f"{stats['total_chunks']} chunks in {duration:.2f}s")
        
        return stats
    
    def query(self, 
              query_text: str, 
              top_k: int = 5,
              min_score: float = 0.3,
              debug: bool = False) -> Dict:
        """
        Query the RAG system
        
        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
            debug: Whether to include debug information
            
        Returns:
            Dictionary containing generated response and metadata
        """
        start_time = datetime.now()
        logger.info(f"Processing query: {query_text}")
        
        # Safety checks on query
        is_safe, reason = self.safety_guard.check_query_safety(query_text)
        if not is_safe:
            logger.warning(f"Query failed safety check: {reason}")
            return {
                'success': False,
                'error': reason,
                'query': query_text
            }
        
        # Retrieve relevant chunks
        logger.info(f"Retrieving top {top_k} chunks...")
        retrieved_chunks = self.retriever.retrieve(
            query_text, 
            top_k=top_k,
            min_score=min_score
        )
        
        if not retrieved_chunks:
            logger.warning("No relevant chunks found")
            return {
                'success': False,
                'error': 'No relevant context found. Please provide more information or try a different query.',
                'query': query_text,
                'retrieved_chunks': 0
            }
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Check evidence quality
        evidence_sufficient, evidence_reason = self.safety_guard.check_evidence_quality(
            retrieved_chunks, 
            min_chunks=2
        )
        
        if not evidence_sufficient:
            logger.warning(f"Insufficient evidence: {evidence_reason}")
            return {
                'success': False,
                'error': evidence_reason,
                'query': query_text,
                'retrieved_chunks': len(retrieved_chunks),
                'suggestions': self._get_clarifying_questions(query_text, retrieved_chunks)
            }
        
        # Generate use cases
        logger.info("Generating use cases...")
        generation_result = self.generator.generate_use_cases(
            query_text, 
            retrieved_chunks
        )
        
        # Check for hallucinations
        is_grounded, hallucination_reason = self.safety_guard.check_hallucination(
            generation_result.get('use_cases', []),
            retrieved_chunks
        )
        
        if not is_grounded:
            logger.warning(f"Hallucination detected: {hallucination_reason}")
            generation_result['warnings'] = [hallucination_reason]
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = {
            'success': True,
            'query': query_text,
            'use_cases': generation_result.get('use_cases', []),
            'assumptions': generation_result.get('assumptions', []),
            'missing_info': generation_result.get('missing_info', []),
            'metadata': {
                'retrieved_chunks': len(retrieved_chunks),
                'duration_seconds': duration,
                'top_k': top_k,
                'min_score': min_score
            }
        }
        
        if debug:
            response['debug'] = {
                'chunks': [
                    {
                        'content': chunk.content[:200] + '...',
                        'source': chunk.metadata.get('source', 'unknown'),
                        'score': chunk.metadata.get('score', 0)
                    }
                    for chunk in retrieved_chunks
                ],
                'warnings': generation_result.get('warnings', [])
            }
        
        logger.info(f"Query completed in {duration:.2f}s")
        return response
    
    def _get_clarifying_questions(self, query: str, chunks: List) -> List[str]:
        """Generate clarifying questions based on query and available context"""
        questions = []
        
        if 'signup' in query.lower() or 'registration' in query.lower():
            if not any('email' in c.content.lower() for c in chunks):
                questions.append("What email validation rules should be applied?")
            if not any('password' in c.content.lower() for c in chunks):
                questions.append("What are the password requirements?")
        
        if 'api' in query.lower() and not any('endpoint' in c.content.lower() for c in chunks):
            questions.append("Which API endpoints are involved?")
        
        return questions or ["Could you provide more context about the feature?"]


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Multimodal RAG System for Test Case Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents from a folder')
    ingest_parser.add_argument('folder', help='Path to folder containing documents')
    ingest_parser.add_argument('--force', action='store_true', help='Force reindexing')
    ingest_parser.add_argument('--config', help='Path to config file')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    query_parser.add_argument('--min-score', type=float, default=0.3, help='Minimum similarity score')
    query_parser.add_argument('--debug', action='store_true', help='Show debug information')
    query_parser.add_argument('--config', help='Path to config file')
    query_parser.add_argument('--output', help='Output file for JSON results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        app = RAGApplication(config_path=args.config)
        
        if args.command == 'ingest':
            stats = app.ingest_documents(args.folder, force_reindex=args.force)
            print("\n" + "="*60)
            print("INGESTION COMPLETE")
            print("="*60)
            print(f"Total files: {stats['total_files']}")
            print(f"Processed: {stats['processed']}")
            print(f"Failed: {stats['failed']}")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Duration: {stats['duration_seconds']:.2f}s")
            
            if stats['errors']:
                print("\nErrors:")
                for error in stats['errors']:
                    print(f"  - {error}")
        
        elif args.command == 'query':
            result = app.query(
                args.query,
                top_k=args.top_k,
                min_score=args.min_score,
                debug=args.debug
            )
            
            print("\n" + "="*60)
            print("QUERY RESULTS")
            print("="*60)
            print(f"Query: {result['query']}")
            print(f"Success: {result['success']}")
            
            if result['success']:
                print(f"\nGenerated {len(result['use_cases'])} use cases")
                print(f"Retrieved {result['metadata']['retrieved_chunks']} chunks")
                print(f"Duration: {result['metadata']['duration_seconds']:.2f}s")
                
                print("\n" + "="*60)
                print("USE CASES (JSON)")
                print("="*60)
                print(json.dumps(result['use_cases'], indent=2))
                
                if result.get('assumptions'):
                    print("\n" + "="*60)
                    print("ASSUMPTIONS")
                    print("="*60)
                    for assumption in result['assumptions']:
                        print(f"  - {assumption}")
                
                if result.get('missing_info'):
                    print("\n" + "="*60)
                    print("MISSING INFORMATION")
                    print("="*60)
                    for info in result['missing_info']:
                        print(f"  - {info}")
                
                if args.debug and result.get('debug'):
                    print("\n" + "="*60)
                    print("DEBUG INFO")
                    print("="*60)
                    for i, chunk in enumerate(result['debug']['chunks'], 1):
                        print(f"\nChunk {i} (score: {chunk['score']:.3f}):")
                        print(f"  Source: {chunk['source']}")
                        print(f"  Content: {chunk['content']}")
            else:
                print(f"\nError: {result.get('error')}")
                if result.get('suggestions'):
                    print("\nSuggestions:")
                    for suggestion in result['suggestions']:
                        print(f"  - {suggestion}")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
