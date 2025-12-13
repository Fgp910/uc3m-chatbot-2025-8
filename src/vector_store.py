"""
Vector Store - Loads ChromaDB from Person B pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

# Load from .env or use default relative path
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./output/chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sgia_chunks")


def get_vectorstore(chromadb_path: str = None):
    path = chromadb_path or CHROMADB_PATH
    
    if not Path(path).exists():
        raise FileNotFoundError(f"ChromaDB not found at: {path}")
    
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=path, 
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    
    print(f"Loaded {vectorstore._collection.count()} chunks")
    return vectorstore


def get_retriever(k_docs: int = 10, filters: dict = None, chromadb_path: str = None):
    vectorstore = get_vectorstore(chromadb_path)
    search_kwargs = {"k": k_docs}
    
    if filters:
        if len(filters) == 1:
            key, val = list(filters.items())[0]
            search_kwargs["filter"] = {key: {"$eq": val}}
        else:
            search_kwargs["filter"] = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
    
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def extract_filters_from_query(query: str) -> dict:
    filters = {}
    q = query.lower()
    
    if 'coast' in q or 'houston' in q:
        filters['zone'] = 'COAST'
    elif 'west' in q:
        filters['zone'] = 'WEST'
    elif 'north' in q:
        filters['zone'] = 'NORTH'
    elif 'south' in q:
        filters['zone'] = 'SOUTH'
    elif 'panhandle' in q:
        filters['zone'] = 'PANHANDLE'
    
    if 'battery' in q or 'storage' in q or 'bess' in q:
        filters['fuel_type'] = 'OTH'
    elif 'solar' in q:
        filters['fuel_type'] = 'SOL'
    elif 'wind' in q:
        filters['fuel_type'] = 'WIN'
    elif 'gas' in q:
        filters['fuel_type'] = 'GAS'
    
    if 'nextera' in q:
        filters['parent_company'] = 'NEXTERA'
    elif 'rwe' in q:
        filters['parent_company'] = 'RWE'
    
    return filters


def get_smart_retriever(query: str, k_docs: int = 15, chromadb_path: str = None):
    filters = extract_filters_from_query(query)
    return get_retriever(k_docs=k_docs, filters=filters if filters else None, chromadb_path=chromadb_path)
