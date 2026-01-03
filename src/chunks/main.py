#!/usr/bin/env python3
"""
ERCOT SGIA RAG Pipeline V6 - Main Entry Point

Built on V5.3 extraction + adds production-grade chunking and indexing.

PIPELINE STEPS:
  Step 1: extract → extraction_results.json (structured metadata) [V5.3]
  Step 2: chunk   → chunks.json (full text chunks for RAG) [V6 NEW]
  Step 3: index   → chromadb/ (vector embeddings) [V6 NEW]

V6 NEW FEATURES:
  - Production text extraction with validation
  - Per-page confidence scoring
  - Checkpoint/resume capability
  - Control character detection and cleaning
  - Quality reporting

Usage:
    # Step 1: Extract structured fields
    python -m src.main extract --pdfs ./pdfs --target_csv ./data/enriched.csv --output ./output

    # Step 2: Create chunks for RAG (with Vision OCR fallback)
    python -m src.main chunk --pdfs ./pdfs --results ./output/extraction_results.json --output ./output

    # Step 3: Index in ChromaDB
    python -m src.main index --chunks ./output/chunks.json --output ./output/chromadb

Author: Santiago (UC3M Applied AI)
Date: December 2025
Version: 6.0
"""

import argparse
import json
import csv
import os
from pathlib import Path


def cmd_extract(args):
    """Run smart extraction pipeline."""
    from src.extraction.metadata import MetadataRegistry
    from src.extraction.smart_extractor import SmartExtractor
    
    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: Anthropic API key required")
        print("  Use --api_key YOUR_KEY or set ANTHROPIC_API_KEY environment variable")
        return
    
    # Load metadata from Gold Master CSV
    print("=" * 60)
    print("ERCOT SGIA SMART EXTRACTION PIPELINE V5.3")
    print("=" * 60)
    
    if args.force_vision:
        print("🔍 FORCE VISION MODE ENABLED")
        print("   - Using Sonnet vision for ALL documents")
        print("   - Extracting expanded fields (POI, voltage, dates)")
        print("   - Cost: ~$0.30-0.50 per document")
    
    registry = MetadataRegistry()
    registry.load_enriched_csv(Path(args.target_csv))
    
    # Initialize extractor
    extractor = SmartExtractor(
        api_key=api_key,
        metadata_registry=registry,
        force_vision=args.force_vision,
    )
    
    # Run extraction
    results = extractor.extract_batch(
        pdf_dir=Path(args.pdfs),
        output_dir=Path(args.output),
        limit=args.limit,
    )
    
    print(f"\n✓ Extraction complete!")
    print(f"  Results: {args.output}/extraction_results.json")
    print(f"  Needs review: {args.output}/needs_review.json")


def cmd_analyze(args):
    """Analyze extraction results."""
    with open(args.results) as f:
        results = json.load(f)
    
    n = len(results)
    print(f"\n{'='*60}")
    print("EXTRACTION ANALYSIS (V5.3)")
    print(f"{'='*60}")
    print(f"\nTotal documents: {n}")
    
    # Field coverage - Core fields
    fields = {
        'security_total_usd': 0,
        'tsp_name': 0,
        'parent_company': 0,
        'is_amended': 0,
    }
    
    # V5.3 expanded fields
    expanded_fields = {
        'poi_name': 0,
        'voltage_kv': 0,
        'effective_date': 0,
        'design_milestone': 0,
        'construction_milestone': 0,
    }
    
    for r in results:
        if r.get('security_total_usd', 0) > 0:
            fields['security_total_usd'] += 1
        if r.get('tsp_name'):
            fields['tsp_name'] += 1
        if r.get('parent_company') and r.get('parent_company') != 'UNKNOWN':
            fields['parent_company'] += 1
        if r.get('is_amended'):
            fields['is_amended'] += 1
        # V5.3 fields
        if r.get('poi_name'):
            expanded_fields['poi_name'] += 1
        if r.get('voltage_kv', 0) > 0:
            expanded_fields['voltage_kv'] += 1
        if r.get('effective_date'):
            expanded_fields['effective_date'] += 1
        if r.get('design_milestone'):
            expanded_fields['design_milestone'] += 1
        if r.get('construction_milestone'):
            expanded_fields['construction_milestone'] += 1
    
    print("\nCore Field Coverage:")
    print("-" * 40)
    for field, count in fields.items():
        pct = 100 * count / n if n > 0 else 0
        status = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
        print(f"  {status} {field}: {count}/{n} ({pct:.0f}%)")
    
    print("\nV5.3 Expanded Field Coverage:")
    print("-" * 40)
    for field, count in expanded_fields.items():
        pct = 100 * count / n if n > 0 else 0
        status = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
        print(f"  {status} {field}: {count}/{n} ({pct:.0f}%)")
    
    # Zone distribution
    zone_counts = {}
    for r in results:
        zone = r.get('zone') or 'OTHER'
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    
    print("\nZone Distribution:")
    print("-" * 40)
    for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
        print(f"  {zone}: {count}")
    
    # Parent company distribution
    parent_counts = {}
    for r in results:
        parent = r.get('parent_company') or 'UNKNOWN'
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
    
    print("\nParent Company Distribution (Top 15):")
    print("-" * 40)
    for parent, count in sorted(parent_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {parent}: {count}")
    
    # TSP distribution
    tsp_counts = {}
    for r in results:
        tsp = r.get('tsp_normalized') or r.get('tsp_name') or 'Unknown'
        tsp_counts[tsp] = tsp_counts.get(tsp, 0) + 1
    
    print("\nTSP Distribution:")
    print("-" * 40)
    for tsp, count in sorted(tsp_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tsp}: {count}")
    
    # Security stats
    securities = [r.get('security_total_usd', 0) for r in results if r.get('security_total_usd', 0) > 0]
    if securities:
        print("\nSecurity Amount Stats:")
        print("-" * 40)
        print(f"  Count: {len(securities)}")
        print(f"  Min: ${min(securities):,.0f}")
        print(f"  Max: ${max(securities):,.0f}")
        print(f"  Avg: ${sum(securities)/len(securities):,.0f}")
    
    # Security per kW stats
    spk = [r.get('security_per_kw', 0) for r in results if r.get('security_per_kw', 0) > 0]
    if spk:
        print("\nSecurity per kW Stats:")
        print("-" * 40)
        print(f"  Count: {len(spk)}")
        print(f"  Min: ${min(spk):.2f}/kW")
        print(f"  Max: ${max(spk):.2f}/kW")
        print(f"  Avg: ${sum(spk)/len(spk):.2f}/kW")
    
    # THE CEO QUESTION: Security per kW by Zone
    zone_security = {}
    for r in results:
        zone = r.get('zone') or 'OTHER'
        s = r.get('security_per_kw', 0)
        if s > 0:
            if zone not in zone_security:
                zone_security[zone] = []
            zone_security[zone].append(s)
    
    if zone_security:
        print("\n" + "=" * 40)
        print("CEO QUESTION: Security $/kW by Zone")
        print("=" * 40)
        for zone in ['WEST', 'PANHANDLE', 'COAST', 'NORTH', 'SOUTH', 'CENTRAL', 'OTHER']:
            if zone in zone_security:
                vals = zone_security[zone]
                avg = sum(vals) / len(vals)
                print(f"  {zone}: ${avg:.2f}/kW (n={len(vals)})")
    
    # Security per kW by Parent Company
    parent_security = {}
    for r in results:
        parent = r.get('parent_company') or 'UNKNOWN'
        s = r.get('security_per_kw', 0)
        if s > 0 and parent != 'UNKNOWN':
            if parent not in parent_security:
                parent_security[parent] = []
            parent_security[parent].append(s)
    
    if parent_security:
        print("\nSecurity $/kW by Developer (Top 10):")
        print("-" * 40)
        for parent, vals in sorted(parent_security.items(), key=lambda x: -len(x[1]))[:10]:
            avg = sum(vals) / len(vals)
            print(f"  {parent}: ${avg:.2f}/kW (n={len(vals)})")
    
    # Needs review
    needs_review = [r for r in results if r.get('needs_review')]
    print(f"\nNeeds Manual Review: {len(needs_review)}/{n} ({100*len(needs_review)/n:.0f}%)")
    
    if needs_review:
        reasons = {}
        for r in needs_review:
            for reason in r.get('review_reasons', []):
                reasons[reason] = reasons.get(reason, 0) + 1
        print("Review Reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


def cmd_export(args):
    """Export results to CSV."""
    with open(args.results) as f:
        results = json.load(f)
    
    columns = [
        'inr', 'item_number', 'project_name', 'developer_spv', 'parent_company',
        'capacity_mw', 'county', 'zone', 'fuel_type', 'technology',
        'tsp_name', 'tsp_normalized',
        'security_design_usd', 'security_construction_usd', 'security_total_usd',
        'security_per_kw',
        # V5.3 expanded fields
        'poi_name', 'voltage_kv', 'effective_date',
        'design_milestone', 'construction_milestone', 'operation_milestone',
        # Parent intel
        'ir_submitter', 'parent_company_inferred',
        # Status
        'is_amended', 'amendment_type',
        'commercial_operation_date',
        'extraction_confidence', 'needs_review',
    ]
    
    output_file = Path(args.output)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"✓ Exported to {output_file}")
    print(f"  {len(results)} rows, {len(columns)} columns")


def cmd_chunk(args):
    """
    STEP 2: Extract full text and create chunks for RAG.
    
    PRODUCTION FEATURES:
    - Per-page validation (detect garbage/OCR failures)
    - Confidence scoring (0.0 to 1.0)
    - Checkpoint/resume (won't re-process on crash)
    - Quality reporting (know what failed)
    - Control character cleaning
    - Retry logic with exponential backoff
    - Output validation with coverage metrics
    
    This does:
    1. Extract full text from each PDF (PyMuPDF + Vision for image pages)
    2. Validate text quality per page (flag garbage for Vision OCR)
    3. Clean control characters and garbage
    4. Split into semantic chunks by section
    5. Attach metadata from extraction_results.json (49 fields)
    6. Validate output quality
    7. Output chunks.json ready for ChromaDB
    """
    from src.chunking import ProductionTextExtractor, SGIAChunker
    from src.chunking.chunker import validate_chunks_output, load_extraction_metadata
    
    # Get API key for Vision OCR
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: Anthropic API key required for Vision OCR")
        print("  Use --api_key YOUR_KEY or set ANTHROPIC_API_KEY environment variable")
        return
    
    print("=" * 70)
    print("STEP 2: PRODUCTION TEXT EXTRACTION + CHUNKING (V6)")
    print("=" * 70)
    
    pdf_dir = Path(args.pdfs)
    output_dir = Path(args.output)
    results_path = Path(args.results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "extracted_text"
    chunks_path = output_dir / "chunks.json"
    checkpoint_dir = output_dir / "checkpoints"
    
    # Phase 1: Extract text from PDFs with validation
    print("\n📄 PHASE 1: Extracting text from PDFs (with validation)...")
    print("-" * 70)
    
    extractor = ProductionTextExtractor(
        api_key=api_key,
        checkpoint_dir=checkpoint_dir,
        enable_checkpoint=True
    )
    
    documents, report = extractor.extract_batch(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        limit=args.limit,
        force=args.force
    )
    
    # Phase 2: Chunk with metadata
    print("\n📦 PHASE 2: Creating chunks with metadata...")
    print("-" * 70)
    
    chunker = SGIAChunker(results_path)
    chunks = chunker.chunk_all(
        text_dir=text_dir,
        output_path=chunks_path,
        limit=args.limit
    )
    
    # Phase 3: Validate output
    print("\n🔍 PHASE 3: Validating output quality...")
    print("-" * 70)
    
    extraction_metadata = load_extraction_metadata(results_path)
    validation = validate_chunks_output(chunks, extraction_metadata)
    
    # Save validation report
    validation_path = output_dir / "chunk_validation.json"
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2)
    
    # Print stats
    stats = chunker.get_stats(chunks)
    
    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)
    print(f"  Total chunks: {validation['total_chunks']}")
    print(f"  Total documents: {validation['total_documents']}")
    print(f"  Avg chunk size: {validation['avg_chunk_size']:.0f} chars")
    print(f"  Range: {validation['min_chunk_size']}-{validation['max_chunk_size']} chars")
    
    print(f"\n  Metadata coverage:")
    for field, pct in validation['coverage'].items():
        status = "✅" if pct >= 90 else "⚠️" if pct >= 70 else "❌"
        print(f"    {status} {field}: {pct:.0f}%")
    
    print(f"\n  Top sections:")
    for section, count in sorted(validation['sections'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {section}: {count}")
    
    if validation['issues']:
        print(f"\n⚠️  ISSUES DETECTED:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    else:
        print(f"\n✅ VALIDATION PASSED")
    
    print(f"\n📁 Output files:")
    print(f"   Text files: {text_dir}/")
    print(f"   Chunks: {chunks_path}")
    print(f"   Extraction report: {output_dir}/extraction_report.json")
    print(f"   Chunk validation: {validation_path}")
    
    if report.documents_needing_review:
        print(f"\n⚠️  {len(report.documents_needing_review)} documents need review:")
        print(f"   Check {output_dir}/extraction_summary.json")
    
    print(f"\n🚀 Next: Run 'python -m src.main index' to create ChromaDB")


def cmd_index(args):
    """
    STEP 3: Create ChromaDB vector embeddings.
    
    This does:
    1. Load chunks.json from Step 2
    2. Create embeddings using sentence-transformers
    3. Store in ChromaDB for semantic search
    """
    from src.indexing import ChromaDBIndexer
    
    print("=" * 70)
    print("STEP 3: CHROMADB INDEXING (V6)")
    print("=" * 70)
    
    chunks_path = Path(args.chunks)
    output_dir = Path(args.output)
    
    # Create indexer
    indexer = ChromaDBIndexer(persist_dir=output_dir)
    
    # Index chunks
    count = indexer.index_chunks(
        chunks_path=chunks_path,
        collection_name=args.collection or "sgia_chunks",
        overwrite=args.overwrite
    )
    
    # Get stats
    stats = indexer.get_stats()
    
    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"  Total chunks indexed: {stats['total_chunks']}")
    print(f"  ChromaDB location: {output_dir}")
    
    # Test query if requested
    if args.test_query:
        print(f"\n🔍 Test query: '{args.test_query}'")
        results = indexer.query(args.test_query, n_results=3)
        for r in results:
            print(f"\n  [{r['chunk_id']}] (dist: {r['distance']:.3f})")
            print(f"  {r['text'][:200]}...")
    
    print(f"\n✅ Ready for RAG! Use ChromaDBIndexer to query.")


def main():
    parser = argparse.ArgumentParser(
        description='ERCOT SGIA RAG Pipeline V6',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PIPELINE STEPS:
  Step 1: extract → extraction_results.json (structured metadata)
  Step 2: chunk   → chunks.json (full text chunks for RAG)  
  Step 3: index   → chromadb/ (vector embeddings for semantic search)

Examples:
  # STEP 1: Extract structured fields from PDFs
  python -m src.main extract --pdfs ./pdfs --target_csv ./data/enriched.csv --output ./output --limit 10
  
  # STEP 2: Extract full text + create chunks for RAG
  python -m src.main chunk --pdfs ./pdfs --results ./output/extraction_results.json --output ./output --limit 5
  
  # STEP 3: Index chunks in ChromaDB
  python -m src.main index --chunks ./output/chunks.json --output ./output/chromadb
  
  # Analyze results
  python -m src.main analyze --results ./output/extraction_results.json
  
  # Export to CSV
  python -m src.main export --results ./output/extraction_results.json --output ./results.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command (Step 1)
    p_extract = subparsers.add_parser('extract', help='Step 1: Extract structured fields from PDFs')
    p_extract.add_argument('--pdfs', required=True, help='Directory containing PDFs')
    p_extract.add_argument('--target_csv', required=True, help='Enriched Gold Master CSV')
    p_extract.add_argument('--output', required=True, help='Output directory')
    p_extract.add_argument('--api_key', help='Anthropic API key')
    p_extract.add_argument('--limit', type=int, help='Limit PDFs (for testing)')
    p_extract.add_argument('--force-vision', action='store_true', dest='force_vision',
                          help='Use vision mode for ALL documents (better accuracy, higher cost)')
    
    # Chunk command (Step 2)
    p_chunk = subparsers.add_parser('chunk', help='Step 2: Extract full text and create chunks for RAG')
    p_chunk.add_argument('--pdfs', required=True, help='Directory containing PDFs')
    p_chunk.add_argument('--results', required=True, help='extraction_results.json from Step 1')
    p_chunk.add_argument('--output', required=True, help='Output directory')
    p_chunk.add_argument('--api_key', help='Anthropic API key (for Vision OCR)')
    p_chunk.add_argument('--limit', type=int, help='Limit PDFs (for testing)')
    p_chunk.add_argument('--force', action='store_true', help='Re-extract all PDFs (ignore checkpoint)')
    
    # Index command (Step 3)
    p_index = subparsers.add_parser('index', help='Step 3: Create ChromaDB vector embeddings')
    p_index.add_argument('--chunks', required=True, help='chunks.json from Step 2')
    p_index.add_argument('--output', required=True, help='ChromaDB output directory')
    p_index.add_argument('--collection', default='sgia_chunks', help='ChromaDB collection name')
    p_index.add_argument('--test-query', dest='test_query', help='Run a test query after indexing')
    p_index.add_argument('--overwrite', action='store_true', help='Recreate collection from scratch')
    
    # Analyze command
    p_analyze = subparsers.add_parser('analyze', help='Analyze extraction results')
    p_analyze.add_argument('--results', required=True, help='extraction_results.json')
    
    # Export command
    p_export = subparsers.add_parser('export', help='Export results to CSV')
    p_export.add_argument('--results', required=True, help='extraction_results.json')
    p_export.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        cmd_extract(args)
    elif args.command == 'chunk':
        cmd_chunk(args)
    elif args.command == 'index':
        cmd_index(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'export':
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
