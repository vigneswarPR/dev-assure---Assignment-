from src.retrieval.retriever import HybridRetriever
from src.utils.config import Config
from collections import Counter

config = Config()
retriever = HybridRetriever(config)
stats = retriever.get_stats()

print(f'Total chunks: {stats["total_chunks"]}')

# Get all chunk sources
sources = []
for chunk in retriever.chunk_map.values():
    sources.append(chunk.metadata.get('filename', 'unknown'))

source_counts = Counter(sources)
print('\nChunk sources:')
for source, count in sorted(source_counts.items()):
    print(f'  {source}: {count} chunks')

# Check if any PNG files are present
png_files = [s for s in sources if s.endswith('.png')]
if png_files:
    print(f'\nPNG files found: {len(set(png_files))} unique files')
    print('PNG sources:', set(png_files))
else:
    print('\nNo PNG files found in chunks')
