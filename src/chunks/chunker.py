"""
Chunker - Section-aware chunking with metadata enrichment

Splits SGIA documents into semantic chunks and attaches metadata
from extraction_results.json for ChromaDB filtering.

Chunk Schema:
{
    "chunk_id": "35077_1330_exhibit_e_001",
    "text": "The Security Amount shall be...",
    "metadata": {
        "inr": "20INR0097",
        "project_name": "El Sauz Ranch",
        "tsp_normalized": "AEP",
        "zone": "WEST",
        "fuel_type": "WIN",
        "security_total_usd": 7500000,
        "parent_company": "APEX",
        "section": "exhibit_e",
        "page_start": 45,
        "chunk_index": 1
    }
}

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chunking configuration
MIN_CHUNK_SIZE = 500      # Minimum characters per chunk
MAX_CHUNK_SIZE = 1500     # Maximum characters per chunk
TARGET_CHUNK_SIZE = 1000  # Ideal chunk size
OVERLAP_SIZE = 100        # Character overlap between chunks


@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    chunk_id: str
    text: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }


# =============================================================================
# SECTION DETECTION
# =============================================================================

# Section patterns in SGIAs
SECTION_PATTERNS = [
    (r'ARTICLE\s+(\d+|[IVX]+)', 'article'),
    (r'EXHIBIT\s+([A-Z])', 'exhibit'),
    (r'APPENDIX\s+([A-Z0-9]+)', 'appendix'),
    (r'SCHEDULE\s+([A-Z0-9]+)', 'schedule'),
    (r'ATTACHMENT\s+([A-Z0-9]+)', 'attachment'),
    (r'RECITALS', 'recitals'),
    (r'WITNESSETH', 'witnesseth'),
]

# High-value sections for competitive intelligence
HIGH_VALUE_SECTIONS = {
    'exhibit_a': 'facility_description',
    'exhibit_b': 'milestones_timeline', 
    'exhibit_c': 'equipment_specs',
    'exhibit_d': 'insurance_requirements',
    'exhibit_e': 'security_amounts',
    'recitals': 'parties_background',
    'article_5': 'interconnection_facilities',
    'article_11': 'security_arrangements',
}


def detect_sections(text: str) -> List[Tuple[int, str, str]]:
    """
    Detect section boundaries in document text.
    
    Returns:
        List of (char_position, section_type, section_id)
    """
    sections = []
    
    for pattern, section_type in SECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos = match.start()
            if section_type in ['article', 'exhibit', 'appendix', 'schedule', 'attachment']:
                section_id = f"{section_type}_{match.group(1).lower()}"
            else:
                section_id = section_type
            sections.append((pos, section_type, section_id))
    
    # Sort by position
    sections.sort(key=lambda x: x[0])
    
    return sections


def split_into_sections(text: str) -> Dict[str, str]:
    """
    Split document into named sections.
    
    Returns:
        Dict mapping section_id -> section_text
    """
    sections_found = detect_sections(text)
    
    if not sections_found:
        # No sections detected, treat as single document
        return {"full_document": text}
    
    result = {}
    
    for i, (pos, section_type, section_id) in enumerate(sections_found):
        # Get text until next section or end
        if i + 1 < len(sections_found):
            end_pos = sections_found[i + 1][0]
        else:
            end_pos = len(text)
        
        section_text = text[pos:end_pos].strip()
        
        # Only keep sections with meaningful content
        if len(section_text) > 100:
            result[section_id] = section_text
    
    return result


# =============================================================================
# CHUNKING LOGIC
# =============================================================================

def chunk_text(
    text: str, 
    min_size: int = MIN_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
    target_size: int = TARGET_CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE
) -> List[str]:
    """
    Split text into chunks of appropriate size.
    
    Strategy:
    1. Try to split on paragraph boundaries
    2. Fall back to sentence boundaries
    3. Last resort: hard split at max_size
    """
    if len(text) <= max_size:
        return [text] if len(text) >= min_size else []
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Determine chunk end
        end_pos = min(current_pos + max_size, len(text))
        
        if end_pos < len(text):
            # Try to find a good break point
            chunk_candidate = text[current_pos:end_pos]
            
            # Prefer paragraph break
            para_break = chunk_candidate.rfind('\n\n')
            if para_break > min_size:
                end_pos = current_pos + para_break
            else:
                # Try sentence break
                sentence_break = max(
                    chunk_candidate.rfind('. '),
                    chunk_candidate.rfind('.\n'),
                    chunk_candidate.rfind('? '),
                    chunk_candidate.rfind('! ')
                )
                if sentence_break > min_size:
                    end_pos = current_pos + sentence_break + 1
        
        # Extract chunk
        chunk = text[current_pos:end_pos].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)
        
        # Move position with overlap
        current_pos = end_pos - overlap if end_pos < len(text) else end_pos
    
    return chunks


# =============================================================================
# METADATA LOADER
# =============================================================================

def load_extraction_metadata(results_path: Path) -> Dict[str, Dict]:
    """
    Load metadata from extraction_results.json.
    
    Returns:
        Dict mapping filename -> metadata dict
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Index by filename
    metadata_by_file = {}
    for r in results:
        filename = r.get('filename', '')
        if filename:
            metadata_by_file[filename] = r
    
    logger.info(f"Loaded metadata for {len(metadata_by_file)} documents")
    return metadata_by_file


def get_chunk_metadata(
    filename: str,
    section_id: str,
    chunk_index: int,
    extraction_metadata: Dict[str, Dict]
) -> Dict:
    """
    Build metadata dict for a chunk.
    
    Combines extraction results with chunk-specific info.
    Supports merged_extraction_results_v5_FINAL.json fields.
    """
    # Get extraction metadata for this file
    file_meta = extraction_metadata.get(filename, {})
    
    # Warn if no metadata found (Bug #4 fix)
    if not file_meta:
        logger.warning(f"No metadata found for {filename} - chunks will have empty metadata")
    
    # Compute queryable security (actual OR estimated)
    security_actual = file_meta.get("security_total_usd", 0) or 0
    security_estimated = file_meta.get("security_estimated_usd", 0) or 0
    security_queryable = security_actual if security_actual > 0 else security_estimated
    
    # Build chunk metadata (include fields useful for filtering)
    metadata = {
        # Primary identifiers
        "inr": file_meta.get("inr", ""),
        "item_number": file_meta.get("item_number", ""),
        "project_name": file_meta.get("project_name", ""),
        
        # Developer info
        "developer_spv": file_meta.get("developer_spv", ""),
        "parent_company": file_meta.get("parent_company", ""),
        "parent_company_inferred": file_meta.get("parent_company_inferred", ""),
        "ir_submitter": file_meta.get("ir_submitter", ""),
        
        # Project specs
        "capacity_mw": file_meta.get("capacity_mw", 0),
        "county": file_meta.get("county", ""),
        "zone": file_meta.get("zone", ""),
        "fuel_type": file_meta.get("fuel_type", ""),
        "technology": file_meta.get("technology", ""),
        "tsp_normalized": file_meta.get("tsp_normalized", ""),
        
        # Security amounts - QUERYABLE FIELD for Person C
        "security_total_usd": security_actual,
        "security_estimated_usd": security_estimated,
        "security_queryable_usd": security_queryable,  # USE THIS FOR QUERIES
        "security_is_estimated": security_actual == 0 and security_estimated > 0,
        "security_per_kw": file_meta.get("security_per_kw", 0),
        "security_design_usd": file_meta.get("security_design_usd", 0),
        "security_construction_usd": file_meta.get("security_construction_usd", 0),
        
        # Amendment status
        "is_amended": file_meta.get("is_amended", False),
        "amendment_type": file_meta.get("amendment_type", ""),
        
        # V5.3 expanded fields
        "poi_name": file_meta.get("poi_name", ""),
        "voltage_kv": file_meta.get("voltage_kv", 0),
        "effective_date": file_meta.get("effective_date", ""),
        
        # Chunk-specific
        "section": section_id,
        "section_type": HIGH_VALUE_SECTIONS.get(section_id, "other"),
        "chunk_index": chunk_index,
        "source_file": filename,
    }
    
    return metadata


# =============================================================================
# MAIN CHUNKER
# =============================================================================

class SGIAChunker:
    """
    Chunk SGIA documents for RAG indexing.
    
    Usage:
        chunker = SGIAChunker(extraction_results_path)
        chunks = chunker.chunk_document(text, filename)
        chunker.chunk_all(text_dir, output_path)
    """
    
    def __init__(self, extraction_results_path: Path):
        """
        Initialize chunker with extraction metadata.
        
        Args:
            extraction_results_path: Path to extraction_results.json
        """
        self.extraction_metadata = load_extraction_metadata(extraction_results_path)
        self.total_chunks = 0
    
    def chunk_document(self, text: str, filename: str) -> List[Chunk]:
        """
        Chunk a single document.
        
        Args:
            text: Full document text
            filename: PDF filename (for metadata lookup)
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Split into sections
        sections = split_into_sections(text)
        
        # Get base identifier from filename (e.g., "35077_1330" from "35077_1330_1234567.pdf")
        base_id = "_".join(filename.replace(".pdf", "").replace(".txt", "").split("_")[:2])
        
        for section_id, section_text in sections.items():
            # Chunk this section
            text_chunks = chunk_text(section_text)
            
            for i, chunk_content in enumerate(text_chunks):
                chunk_id = f"{base_id}_{section_id}_{i:03d}"
                
                metadata = get_chunk_metadata(
                    filename=filename.replace(".txt", ".pdf"),  # Match extraction filename
                    section_id=section_id,
                    chunk_index=i,
                    extraction_metadata=self.extraction_metadata
                )
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    metadata=metadata
                ))
        
        return chunks
    
    def chunk_all(
        self, 
        text_dir: Path, 
        output_path: Path,
        limit: Optional[int] = None
    ) -> List[Chunk]:
        """
        Chunk all extracted text files.
        
        Args:
            text_dir: Directory containing .txt files from text extraction
            output_path: Where to save chunks.json
            limit: Max documents to process (for testing)
            
        Returns:
            List of all chunks
        """
        # Find all text files
        txt_files = sorted(text_dir.glob("*.txt"))
        if limit:
            txt_files = txt_files[:limit]
        
        logger.info(f"Chunking {len(txt_files)} documents...")
        
        all_chunks = []
        
        for i, txt_path in enumerate(txt_files):
            logger.info(f"[{i+1}/{len(txt_files)}] {txt_path.name}")
            
            try:
                # Read text
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Skip empty or very short files (Bug #6 fix)
                if len(text.strip()) < 100:
                    logger.warning(f"  Skipping - too short ({len(text)} chars)")
                    continue
                
                # Chunk it
                doc_chunks = self.chunk_document(text, txt_path.name)
                all_chunks.extend(doc_chunks)
                
                logger.info(f"  → {len(doc_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue
        
        self.total_chunks = len(all_chunks)
        
        # Save chunks
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2)
        
        logger.info(f"\nChunking complete:")
        logger.info(f"  Documents: {len(txt_files)}")
        logger.info(f"  Total chunks: {self.total_chunks}")
        logger.info(f"  Output: {output_path}")
        
        # Stats
        if all_chunks:
            avg_size = sum(len(c.text) for c in all_chunks) / len(all_chunks)
            logger.info(f"  Avg chunk size: {avg_size:.0f} chars")
        
        return all_chunks
    
    def get_stats(self, chunks: List[Chunk]) -> Dict:
        """Get statistics about chunks."""
        if not chunks:
            return {}
        
        sections = {}
        zones = {}
        fuel_types = {}
        
        for c in chunks:
            # Section distribution
            section = c.metadata.get("section", "unknown")
            sections[section] = sections.get(section, 0) + 1
            
            # Zone distribution
            zone = c.metadata.get("zone", "unknown")
            zones[zone] = zones.get(zone, 0) + 1
            
            # Fuel type distribution
            fuel = c.metadata.get("fuel_type", "unknown")
            fuel_types[fuel] = fuel_types.get(fuel, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.text) for c in chunks) / len(chunks),
            "sections": dict(sorted(sections.items(), key=lambda x: -x[1])),
            "zones": dict(sorted(zones.items(), key=lambda x: -x[1])),
            "fuel_types": dict(sorted(fuel_types.items(), key=lambda x: -x[1])),
        }


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_chunks_output(chunks: List[Chunk], extraction_metadata: Dict) -> Dict:
    """
    Validate chunking output quality.
    
    Returns:
        Dict with validation metrics and issues
    """
    issues = []
    
    # Basic counts
    total_chunks = len(chunks)
    total_docs = len(set(c.metadata.get('source_file', '') for c in chunks))
    
    # Metadata coverage
    fields_to_check = [
        'inr', 'project_name', 'zone', 'fuel_type', 
        'security_queryable_usd', 'tsp_normalized'
    ]
    
    coverage = {field: 0 for field in fields_to_check}
    for c in chunks:
        for field in fields_to_check:
            val = c.metadata.get(field)
            if val and val != "" and val != 0:
                coverage[field] += 1
    
    # Calculate percentages
    coverage_pct = {k: v / total_chunks * 100 for k, v in coverage.items()}
    
    # Check for issues
    if coverage_pct.get('zone', 0) < 90:
        issues.append(f"Low zone coverage: {coverage_pct['zone']:.0f}%")
    if coverage_pct.get('fuel_type', 0) < 90:
        issues.append(f"Low fuel_type coverage: {coverage_pct['fuel_type']:.0f}%")
    if coverage_pct.get('security_queryable_usd', 0) < 80:
        issues.append(f"Low security coverage: {coverage_pct['security_queryable_usd']:.0f}%")
    
    # Chunk size analysis
    chunk_sizes = [len(c.text) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    if avg_size < 200:
        issues.append(f"Chunks too small: avg {avg_size:.0f} chars")
    if avg_size > 2000:
        issues.append(f"Chunks too large: avg {avg_size:.0f} chars")
    
    # Section distribution
    sections = {}
    for c in chunks:
        section = c.metadata.get('section', 'unknown')
        sections[section] = sections.get(section, 0) + 1
    
    # Documents without chunks
    docs_in_meta = set(extraction_metadata.keys())
    docs_in_chunks = set(c.metadata.get('source_file', '') for c in chunks)
    missing_docs = docs_in_meta - docs_in_chunks
    
    if missing_docs:
        issues.append(f"{len(missing_docs)} documents have no chunks")
    
    return {
        'total_chunks': total_chunks,
        'total_documents': total_docs,
        'avg_chunk_size': avg_size,
        'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
        'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
        'coverage': coverage_pct,
        'sections': sections,
        'missing_documents': list(missing_docs)[:10],  # First 10 only
        'issues': issues,
        'passed': len(issues) == 0
    }
