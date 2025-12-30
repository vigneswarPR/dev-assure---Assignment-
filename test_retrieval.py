"""
Test retrieval functionality without needing LLM API
Tests chunking, indexing, and hybrid retrieval
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from src.retrieval.retriever import HybridRetriever
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_retrieval(query: str, top_k: int = 5, min_score: float = 0.3, use_hyde: bool = False):
    """
    Test retrieval without LLM
    
    Args:
        query: Search query
        top_k: Number of results
        min_score: Minimum similarity threshold
    """
    print("\n" + "="*70)
    print("RETRIEVAL TEST (No LLM Required)")
    print("="*70)
    
    # Initialize
    config = Config()
    retriever = HybridRetriever(config)
    
    # Get stats
    stats = retriever.get_stats()
    print(f"\n[*] Index Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   BM25 enabled: {stats['bm25_enabled']}")

    if stats['total_chunks'] == 0:
        print("\n[ERROR] No documents indexed!")
        print("   Run: python app.py ingest ./sample_data")
        return
    
    # Perform retrieval
    print(f"\n[SEARCH] Query: '{query}'")
    print(f"   Parameters: top_k={top_k}, min_score={min_score}")
    print("\n" + "-"*70)

    chunks = retriever.retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
        use_hybrid=True,
        use_hyde=use_hyde
    )

    if not chunks:
        print("\n[ERROR] No relevant chunks found!")
        print("   Try:")
        print("   - Lowering min_score (e.g., --min-score 0.2)")
        print("   - Different query terms")
        print("   - Ingesting more documents")
        return

    # Display results
    print(f"\n[SUCCESS] Found {len(chunks)} relevant chunks\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"{'='*70}")
        print(f"CHUNK {i}/{len(chunks)}")
        print(f"{'='*70}")
        
        # Metadata
        print(f"\n[FILE] Source: {chunk.metadata.get('filename', 'unknown')}")
        print(f"[ID] Chunk ID: {chunk.chunk_id}")
        print(f"[LEN] Length: {len(chunk.content)} chars")

        # Scores
        score = chunk.metadata.get('score', 0)
        print(f"\n[SCORES]:")
        print(f"   Overall: {score:.4f}")

        if 'vector_score' in chunk.metadata:
            print(f"   Vector: {chunk.metadata.get('vector_score', 0):.4f}")
        if 'bm25_score' in chunk.metadata:
            print(f"   BM25: {chunk.metadata.get('bm25_score', 0):.4f}")

        print(f"   Method: {chunk.metadata.get('retrieval_method', 'unknown')}")

        if chunk.metadata.get('reranked'):
            print(f"   [RERANKED] Yes")

        # Content preview
        print(f"\n[CONTENT]:")
        print("-"*70)
        
        # Show first 500 chars
        content_preview = chunk.content[:500]
        if len(chunk.content) > 500:
            content_preview += "..."
        
        print(content_preview)
        print()
    
    print("="*70)
    print("RETRIEVAL TEST COMPLETE")
    print("="*70)
    
    # Show relevance distribution
    scores = [c.metadata.get('score', 0) for c in chunks]
    print(f"\n[STATS] Relevance Distribution:")
    print(f"   Highest: {max(scores):.4f}")
    print(f"   Lowest: {min(scores):.4f}")
    print(f"   Average: {sum(scores)/len(scores):.4f}")

    # Suggest improvements
    avg_score = sum(scores) / len(scores)
    if avg_score < 0.4:
        print(f"\n[TIPS]:")
        print(f"   - Low average relevance ({avg_score:.2f})")
        print(f"   - Try more specific queries")
        print(f"   - Ingest more relevant documents")


def test_multiple_queries():
    """Test multiple queries at once"""
    test_queries = [
        "signup registration email",
        "password requirements validation",
        "login authentication",
        "user account creation",
        "email verification"
    ]
    
    print("\n" + "="*70)
    print("TESTING MULTIPLE QUERIES")
    print("="*70)
    
    config = Config()
    retriever = HybridRetriever(config)
    
    stats = retriever.get_stats()
    if stats['total_chunks'] == 0:
        print("\n[ERROR] No documents indexed!")
        return

    results_summary = []

    for query in test_queries:
        print(f"\n[SEARCH] Query: '{query}'")
        chunks = retriever.retrieve(query, top_k=3, min_score=0.3)

        if chunks:
            avg_score = sum(c.metadata.get('score', 0) for c in chunks) / len(chunks)
            print(f"   [SUCCESS] Found {len(chunks)} chunks (avg score: {avg_score:.3f})")
            results_summary.append((query, len(chunks), avg_score))
        else:
            print(f"   [NO RESULTS] No results")
            results_summary.append((query, 0, 0.0))
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Query':<40} {'Results':<10} {'Avg Score':<10}")
    print("-"*70)
    for query, count, score in results_summary:
        print(f"{query:<40} {count:<10} {score:<10.3f}")


def compare_retrieval_methods(query: str):
    """Compare vector-only vs hybrid retrieval"""
    print("\n" + "="*70)
    print("COMPARING RETRIEVAL METHODS")
    print("="*70)

    config = Config()
    retriever = HybridRetriever(config)

    print(f"\n[SEARCH] Query: '{query}'")

    # Test vector-only
    print("\n[1] Vector-Only Retrieval:")
    vector_chunks = retriever.retrieve(query, top_k=5, use_hybrid=False)
    if vector_chunks:
        avg_vector = sum(c.metadata.get('score', 0) for c in vector_chunks) / len(vector_chunks)
        print(f"   Found {len(vector_chunks)} chunks (avg: {avg_vector:.3f})")
        print(f"   Top sources: {', '.join(set(c.metadata.get('filename', '?') for c in vector_chunks[:3]))}")
    else:
        print("   No results")

    # Test hybrid
    print("\n[2] Hybrid Retrieval (Vector + BM25 + RRF):")
    hybrid_chunks = retriever.retrieve(query, top_k=5, use_hybrid=True)
    if hybrid_chunks:
        avg_hybrid = sum(c.metadata.get('score', 0) for c in hybrid_chunks) / len(hybrid_chunks)
        print(f"   Found {len(hybrid_chunks)} chunks (avg: {avg_hybrid:.3f})")
        print(f"   Top sources: {', '.join(set(c.metadata.get('filename', '?') for c in hybrid_chunks[:3]))}")
    else:
        print("   No results")

    # Compare
    if vector_chunks and hybrid_chunks:
        print(f"\n[COMPARISON]:")
        print(f"   Vector avg score: {avg_vector:.3f}")
        print(f"   Hybrid avg score: {avg_hybrid:.3f}")
        if avg_hybrid > avg_vector:
            print(f"   [SUCCESS] Hybrid is better (+{(avg_hybrid - avg_vector):.3f})")
        else:
            print(f"   [WARNING] Vector is better (+{(avg_vector - avg_hybrid):.3f})")


def compare_hyde_vs_standard(query: str):
    """Compare standard retrieval vs HYDE retrieval"""
    print("\n" + "="*70)
    print("COMPARING STANDARD VS HYDE RETRIEVAL")
    print("="*70)

    # Test standard retrieval
    print(f"\n[SEARCH] Query: '{query}'")

    config = Config()
    config.use_hyde = False  # Ensure HYDE is disabled for standard test
    retriever = HybridRetriever(config)

    print("\n[1] Standard Hybrid Retrieval:")
    standard_chunks = retriever.retrieve(query, top_k=5, use_hybrid=True, use_hyde=False)
    if standard_chunks:
        avg_standard = sum(c.metadata.get('score', 0) for c in standard_chunks) / len(standard_chunks)
        print(f"   Found {len(standard_chunks)} chunks (avg: {avg_standard:.3f})")
        print(f"   Top sources: {', '.join(set(c.metadata.get('filename', '?') for c in standard_chunks[:3]))}")
    else:
        print("   No results")

    # Test HYDE retrieval
    print("\n[2] HYDE Retrieval (Hypothetical Document Embeddings):")
    config.use_hyde = True  # Enable HYDE
    hyde_retriever = HybridRetriever(config)  # Reinitialize to pick up config change

    hyde_chunks = hyde_retriever.retrieve(query, top_k=5, use_hybrid=True, use_hyde=True)
    if hyde_chunks:
        avg_hyde = sum(c.metadata.get('score', 0) for c in hyde_chunks) / len(hyde_chunks)
        print(f"   Found {len(hyde_chunks)} chunks (avg: {avg_hyde:.3f})")
        print(f"   Top sources: {', '.join(set(c.metadata.get('filename', '?') for c in hyde_chunks[:3]))}")
    else:
        print("   No results")

    # Compare
    if standard_chunks and hyde_chunks:
        print(f"\n[COMPARISON]:")
        print(f"   Standard avg score: {avg_standard:.3f}")
        print(f"   HYDE avg score: {avg_hyde:.3f}")
        if avg_hyde > avg_standard:
            print(f"   [SUCCESS] HYDE is better (+{(avg_hyde - avg_standard):.3f})")
        else:
            print(f"   [INFO] Standard is better (+{(avg_standard - avg_hyde):.3f})")

        # Show overlap
        standard_ids = set(c.chunk_id for c in standard_chunks)
        hyde_ids = set(c.chunk_id for c in hyde_chunks)
        overlap = len(standard_ids & hyde_ids)
        print(f"   Chunk overlap: {overlap}/{len(standard_chunks)} chunks")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test retrieval without LLM')
    parser.add_argument('query', nargs='?', default='signup email password', 
                       help='Search query')
    parser.add_argument('--top-k', type=int, default=5, 
                       help='Number of results (default: 5)')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum similarity score (default: 0.3)')
    parser.add_argument('--multiple', action='store_true',
                       help='Test multiple queries')
    parser.add_argument('--compare', action='store_true',
                       help='Compare retrieval methods')
    parser.add_argument('--hyde', action='store_true',
                       help='Use HYDE (Hypothetical Document Embeddings)')
    parser.add_argument('--compare-hyde', action='store_true',
                       help='Compare standard vs HYDE retrieval')
    
    args = parser.parse_args()
    
    try:
        if args.multiple:
            test_multiple_queries()
        elif args.compare:
            compare_retrieval_methods(args.query)
        elif args.compare_hyde:
            compare_hyde_vs_standard(args.query)
        else:
            test_retrieval(args.query, args.top_k, args.min_score, args.hyde)
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        print(f"\n[ERROR] Error: {str(e)}")
        sys.exit(1)