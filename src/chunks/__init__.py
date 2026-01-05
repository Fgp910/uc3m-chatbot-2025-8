"""
ERCOT SGIA Document Processing Pipeline

This package contains the complete document ingestion pipeline:
- Text extraction (PyMuPDF + Claude Vision OCR)
- Smart field extraction (Claude Haiku/Sonnet)
- Section-aware chunking for RAG
- ChromaDB indexing with embeddings

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

from .metadata import MetadataRegistry, ProjectMetadata
from .smart_extractor import SmartExtractor
from .text_extractor import ProductionTextExtractor
from .chunker import SGIAChunker, validate_chunks_output, load_extraction_metadata
from .chromadb_index import ChromaDBIndexer
from .validation import ExtractionValidator, validate_extraction
from .parent_extractor import ParentCompanyExtractor, extract_parent_company
from .checkpoint import CheckpointManager

__all__ = [
    # Core classes
    'MetadataRegistry',
    'ProjectMetadata',
    'SmartExtractor',
    'ProductionTextExtractor',
    'SGIAChunker',
    'ChromaDBIndexer',
    'ExtractionValidator',
    'ParentCompanyExtractor',
    'CheckpointManager',
    # Convenience functions
    'validate_chunks_output',
    'load_extraction_metadata',
    'validate_extraction',
    'extract_parent_company',
]
