#!/usr/bin/env python3
"""
Clear the ChromaDB collection to remove old chunks
"""

from src.utils.config import Config
from src.utils.logger import setup_logger
import chromadb
from chromadb.config import Settings
from pathlib import Path

logger = setup_logger(__name__)

def clear_database():
    """Clear all chunks from the ChromaDB collection"""
    config = Config()

    # Initialize ChromaDB
    persist_dir = Path(config.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to ChromaDB at: {persist_dir}")
    chroma_client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete collection if it exists
    try:
        collection = chroma_client.get_collection(name=config.chroma_collection_name)
        chroma_client.delete_collection(name=config.chroma_collection_name)
        logger.info(f"Deleted collection: {config.chroma_collection_name}")
    except Exception as e:
        logger.warning(f"Collection might not exist: {e}")

    # Create fresh collection
    collection = chroma_client.create_collection(
        name=config.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("Database cleared successfully")
    print(" Database cleared - old chunks removed")

if __name__ == "__main__":
    clear_database()

