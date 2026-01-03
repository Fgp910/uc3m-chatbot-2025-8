"""
ChromaDB Indexer - Create vector embeddings for RAG

Loads chunks.json from Step 2 and creates a ChromaDB collection
with semantic embeddings and metadata filters.

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "sgia_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
BATCH_SIZE = 100  # Chunks per batch for embedding


class ChromaDBIndexer:
    """
    Index SGIA chunks in ChromaDB for RAG retrieval.
    
    Usage:
        indexer = ChromaDBIndexer(persist_dir="./chromadb")
        indexer.index_chunks(chunks_path="./chunks.json")
        
        # Query
        results = indexer.query("security requirements for battery projects", 
                               filters={"fuel_type": "OTH"})
    """
    
    def __init__(self, persist_dir: Path, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize indexer.
        
        Args:
            persist_dir: Directory to persist ChromaDB
            embedding_model: Sentence transformer model name
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Track which collection we're using
        self._collection_name = COLLECTION_NAME
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model)
            logger.info(f"  Embedding dimension: {self.embedder.get_sentence_embedding_dimension()}")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        
        self.collection = None
    
    def create_collection(self, name: str = COLLECTION_NAME, overwrite: bool = False):
        """
        Create or get ChromaDB collection.
        
        Args:
            name: Collection name
            overwrite: If True, delete existing collection first
        """
        if overwrite:
            try:
                self.client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"description": "ERCOT SGIA documents for RAG"}
        )
        
        # Store collection name for later use in query/get_stats
        self._collection_name = name
        
        logger.info(f"Collection '{name}' ready. Current count: {self.collection.count()}")
        return self.collection
    
    def index_chunks(
        self, 
        chunks_path: Path,
        collection_name: str = COLLECTION_NAME,
        overwrite: bool = False
    ) -> int:
        """
        Index all chunks from chunks.json.
        
        Args:
            chunks_path: Path to chunks.json from Step 2
            collection_name: ChromaDB collection name
            overwrite: If True, recreate collection from scratch
            
        Returns:
            Number of chunks indexed
        """
        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)
        
        logger.info(f"  Loaded {len(chunks)} chunks")
        
        # Create collection
        self.create_collection(collection_name, overwrite=overwrite)
        
        # Index in batches
        logger.info(f"Indexing chunks (batch size: {BATCH_SIZE})...")
        
        indexed = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            
            # Prepare batch data
            ids = [c['chunk_id'] for c in batch]
            documents = [c['text'] for c in batch]
            metadatas = [self._clean_metadata(c['metadata']) for c in batch]
            
            # Generate embeddings
            embeddings = self.embedder.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            indexed += len(batch)
            logger.info(f"  Indexed {indexed}/{len(chunks)} chunks")
        
        logger.info(f"\n✅ Indexing complete: {indexed} chunks in '{collection_name}'")
        return indexed
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata for ChromaDB (must be str, int, float, or bool).
        """
        cleaned = {}
        for k, v in metadata.items():
            if v is None:
                continue
            elif isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, list):
                cleaned[k] = ",".join(str(x) for x in v)
            else:
                cleaned[k] = str(v)
        return cleaned
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the collection.
        
        Args:
            query_text: Natural language query
            n_results: Number of results to return
            filters: Metadata filters (e.g., {"zone": "WEST", "fuel_type": "SOL"})
            
        Returns:
            List of results with text, metadata, and distance
        """
        if self.collection is None:
            self.collection = self.client.get_collection(self._collection_name)
        
        # Build where clause
        where = None
        if filters:
            if len(filters) == 1:
                key, val = list(filters.items())[0]
                where = {key: val}
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query_text]).tolist()
        
        # Query
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        if self.collection is None:
            self.collection = self.client.get_collection(self._collection_name)
        
        return {
            "collection_name": self.collection.name,
            "total_chunks": self.collection.count(),
            "persist_dir": str(self.persist_dir)
        }


def test_indexer():
    """Quick test of the indexer."""
    # Create test chunks
    test_chunks = [
        {
            "chunk_id": "test_001",
            "text": "The security amount for this solar project is $5,000,000.",
            "metadata": {
                "inr": "20INR0001",
                "project_name": "Test Solar",
                "fuel_type": "SOL",
                "zone": "WEST",
                "security_total_usd": 5000000
            }
        },
        {
            "chunk_id": "test_002", 
            "text": "Battery storage project requires network upgrades costing $2,500,000.",
            "metadata": {
                "inr": "21INR0002",
                "project_name": "Test Battery",
                "fuel_type": "OTH",
                "zone": "COAST",
                "security_total_usd": 2500000
            }
        }
    ]
    
    # Save test chunks
    test_path = Path("/tmp/test_chunks.json")
    with open(test_path, 'w') as f:
        json.dump(test_chunks, f)
    
    # Index
    indexer = ChromaDBIndexer(persist_dir=Path("/tmp/test_chromadb"))
    indexer.index_chunks(test_path, overwrite=True)
    
    # Query
    results = indexer.query("security requirements for solar")
    print("\nQuery: 'security requirements for solar'")
    for r in results:
        print(f"  {r['chunk_id']}: {r['text'][:50]}... (dist: {r['distance']:.3f})")
    
    # Query with filter
    results = indexer.query("security amount", filters={"zone": "COAST"})
    print("\nQuery: 'security amount' (zone=COAST)")
    for r in results:
        print(f"  {r['chunk_id']}: {r['text'][:50]}... (dist: {r['distance']:.3f})")


if __name__ == "__main__":
    test_indexer()
