from src.retrieval.retriever import HybridRetriever
from src.utils.config import Config

retriever = HybridRetriever(Config())
stats = retriever.get_stats()
print(f'Chunks: {stats["total_chunks"]}')

# Check sources
sources = {}
for chunk in retriever.chunk_map.values():
    src = chunk.metadata.get('filename', 'unknown')
    sources[src] = sources.get(src, 0) + 1

print('\nSources:')
for src, count in sorted(sources.items()):
    print(f'  {src}: {count} chunks')

# Check for PNG files
png_sources = [s for s in sources.keys() if s.endswith('.png')]
if png_sources:
    print(f'\n[SUCCESS] PNG files indexed: {len(png_sources)}')
    total_png_chunks = sum(sources[s] for s in png_sources)
    print(f'PNG chunks: {total_png_chunks}')
else:
    print('\n[INFO] No PNG files found in index')
