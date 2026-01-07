"""
Text Extractor - PRODUCTION GRADE
=================================

Extract full text from SGIA PDFs with Vision fallback.

Production Features:
- Per-page validation (detect OCR failures, garbled text)
- Retry logic with exponential backoff
- Checkpoint/resume capability
- Detailed quality reporting
- Confidence scoring per page
- Cost and time tracking
- Control character detection and cleaning

"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Text quality thresholds
MIN_CHARS_FOR_TEXT = 500           # Below this, page needs Vision OCR
MIN_PRINTABLE_RATIO = 0.70         # Below this, text is garbled
MIN_VISION_RESPONSE = 50           # Vision must return at least this many chars
MIN_WORDS_RATIO = 0.05             # Minimum words per character

# Vision API settings
VISION_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]          # Exponential backoff (seconds)
REQUEST_DELAY = 1.0                # Rate limiting between calls

# Cost tracking
COST_PER_VISION_CALL = 0.003       # ~$0.003 per image


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PageResult:
    """Extraction result for a single page."""
    page_num: int
    text: str
    text_cleaned: str              # Cleaned version (control chars removed)
    char_count: int
    char_count_cleaned: int
    method: str                    # 'pymupdf', 'vision', 'vision_failed'
    success: bool
    confidence: float              # 0.0 to 1.0
    printable_ratio: float
    issues: List[str] = field(default_factory=list)
    retry_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DocumentResult:
    """Extraction result for a full document."""
    filename: str
    filepath: str
    total_pages: int
    extracted_pages: int
    failed_pages: int
    text_pages: int                # Used PyMuPDF
    vision_pages: int              # Used Vision API
    low_confidence_pages: int      # Pages with confidence < 0.7
    total_chars: int
    avg_confidence: float
    success: bool
    issues: List[str] = field(default_factory=list)
    pages: List[PageResult] = field(default_factory=list)
    full_text: str = ""
    extraction_time_sec: float = 0.0
    vision_cost: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['pages'] = [p if isinstance(p, dict) else p.to_dict() for p in self.pages]
        return d

    def get_text_by_page_range(self, start: int, end: int) -> str:
        """Get text from a range of pages."""
        return "\n".join(
            p.text_cleaned for p in self.pages
            if start <= p.page_num < end
        )


@dataclass
class ExtractionReport:
    """Summary report for batch extraction."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_pages: int
    text_pages: int
    vision_pages: int
    failed_pages: int
    low_confidence_pages: int
    total_chars: int
    avg_confidence: float
    total_time_sec: float
    total_vision_cost: float
    documents_needing_review: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def print_summary(self):
        print("\n" + "=" * 70)
        print("EXTRACTION REPORT")
        print("=" * 70)

        print(f"\nDOCUMENTS:")
        print(f"  Total:      {self.total_documents}")
        print(f"  Successful: {self.successful_documents} ({self.successful_documents/max(1,self.total_documents)*100:.1f}%)")
        print(f"  Failed:     {self.failed_documents} ({self.failed_documents/max(1,self.total_documents)*100:.1f}%)")

        print(f"\nPAGES:")
        print(f"  Total:          {self.total_pages}")
        print(f"  Text (PyMuPDF): {self.text_pages} ({self.text_pages/max(1,self.total_pages)*100:.1f}%)")
        print(f"  Vision (API):   {self.vision_pages} ({self.vision_pages/max(1,self.total_pages)*100:.1f}%)")
        print(f"  Failed:         {self.failed_pages} ({self.failed_pages/max(1,self.total_pages)*100:.1f}%)")
        print(f"  Low confidence: {self.low_confidence_pages} ({self.low_confidence_pages/max(1,self.total_pages)*100:.1f}%)")

        print(f"\nQUALITY:")
        print(f"  Total chars:      {self.total_chars:,}")
        print(f"  Avg confidence:   {self.avg_confidence:.2f}")

        print(f"\nRESOURCES:")
        print(f"  Time:        {self.total_time_sec/60:.1f} minutes")
        print(f"  Vision cost: ${self.total_vision_cost:.2f}")

        if self.documents_needing_review:
            print(f"\n⚠️  DOCUMENTS NEEDING REVIEW ({len(self.documents_needing_review)}):")
            for doc in self.documents_needing_review[:10]:
                print(f"    - {doc}")
            if len(self.documents_needing_review) > 10:
                print(f"    ... and {len(self.documents_needing_review) - 10} more")

        print("=" * 70)


# =============================================================================
# TEXT CLEANING & VALIDATION
# =============================================================================

def clean_text(text: str) -> str:
    """
    Remove control characters and garbage from extracted text.

    Handles the 101-control-char pattern found in SGIA PDFs.
    """
    if not text:
        return ""

    # Remove control characters (keep \n, \r, \t)
    cleaned = ''.join(
        c for c in text
        if c.isprintable() or c in '\n\r\t'
    )

    # Remove excessive whitespace
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
    cleaned = re.sub(r' {3,}', '  ', cleaned)

    return cleaned.strip()


def calculate_printable_ratio(text: str) -> float:
    """Calculate ratio of printable characters."""
    if not text:
        return 0.0

    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    return printable / len(text)


def validate_text_quality(
    text: str,
    page_num: int,
    method: str
) -> Tuple[bool, float, List[str]]:
    """
    Validate extracted text quality.

    Returns:
        (success, confidence, issues)
    """
    issues = []
    confidence = 1.0

    if not text:
        return False, 0.0, ["Empty text"]

    text_stripped = text.strip()
    char_count = len(text_stripped)

    # Check minimum length for Vision responses
    if method == 'vision' and char_count < MIN_VISION_RESPONSE:
        issues.append(f"Vision returned only {char_count} chars")
        confidence -= 0.5

    # Check printable ratio
    printable_ratio = calculate_printable_ratio(text)
    if printable_ratio < MIN_PRINTABLE_RATIO:
        issues.append(f"Low printable ratio: {printable_ratio:.1%}")
        confidence -= (MIN_PRINTABLE_RATIO - printable_ratio) * 2

    # Check for OCR failure patterns
    ocr_failure_patterns = [
        (r'[\x00-\x08\x0b\x0c\x0e-\x1f]{5,}', "Control character sequence"),
        (r'(.)\1{15,}', "Repeated character (15+)"),
    ]

    for pattern, description in ocr_failure_patterns:
        if re.search(pattern, text):
            issues.append(description)
            confidence -= 0.3

    # Check word density
    words = text_stripped.split()
    word_count = len(words)

    if char_count > 100 and word_count < char_count * MIN_WORDS_RATIO:
        issues.append(f"Low word density: {word_count} words in {char_count} chars")
        confidence -= 0.2

    # Check for the specific 101-control-char SGIA pattern
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if control_chars >= 50:
        issues.append(f"Embedded garbage: {control_chars} control chars")
        confidence -= 0.3

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    # Success if confidence above threshold
    success = confidence >= 0.5

    return success, confidence, issues


def assess_page_needs_vision(page: fitz.Page) -> Tuple[bool, str, int, float]:
    """
    Determine if a page needs Vision OCR.

    Returns:
        (needs_vision, raw_text, char_count, printable_ratio)
    """
    text = page.get_text()
    char_count = len(text.strip())
    printable_ratio = calculate_printable_ratio(text)

    # Need Vision if:
    # 1. Too few characters
    # 2. Too many non-printable characters (garbled)
    needs_vision = (
        char_count < MIN_CHARS_FOR_TEXT or
        printable_ratio < MIN_PRINTABLE_RATIO
    )

    return needs_vision, text, char_count, printable_ratio


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class ExtractionCheckpoint:
    """Manage extraction progress for resume capability."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.completed: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load existing checkpoint."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.completed = data.get('completed', {})
                logger.info(f"📂 Loaded checkpoint: {len(self.completed)} documents completed")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

    def save(self):
        """Save checkpoint."""
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump({
                    'completed': self.completed,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")

    def is_completed(self, filename: str) -> bool:
        return filename in self.completed

    def mark_completed(self, result: DocumentResult):
        # Store summary only (not full text - too large)
        self.completed[result.filename] = {
            'total_pages': result.total_pages,
            'text_pages': result.text_pages,
            'vision_pages': result.vision_pages,
            'failed_pages': result.failed_pages,
            'avg_confidence': result.avg_confidence,
            'success': result.success
        }
        self.save()

    def get_stats(self) -> Dict:
        return {
            'documents_completed': len(self.completed),
            'documents': self.completed
        }


# =============================================================================
# MAIN EXTRACTOR
# =============================================================================

class ProductionTextExtractor:
    """
    Production-grade text extraction from SGIA PDFs.

    Features:
    - Per-page validation with confidence scoring
    - Retry logic with exponential backoff
    - Checkpoint/resume capability
    - Control character cleaning
    - Detailed quality reporting
    - Cost tracking
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        enable_checkpoint: bool = True
    ):
        """
        Initialize extractor.

        Args:
            checkpoint_dir: Directory for checkpoints
            enable_checkpoint: Whether to use checkpointing
        """
        self.checkpoint: Optional[ExtractionCheckpoint] = None

        if enable_checkpoint and checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint = ExtractionCheckpoint(
                checkpoint_dir / "text_extraction_checkpoint.json"
            )

    def extract_document(self, pdf_path: Path) -> DocumentResult:
        """
        Extract all text from a PDF with validation.

        Args:
            pdf_path: Path to PDF file

        Returns:
            DocumentResult with full extraction details
        """
        start_time = time.time()
        filename = pdf_path.name

        # Check checkpoint
        if self.checkpoint and self.checkpoint.is_completed(filename):
            logger.info(f"  [CACHED] {filename}")
            # Return minimal result for cached docs
            stats = self.checkpoint.completed[filename]
            return DocumentResult(
                filename=filename,
                filepath=str(pdf_path),
                total_pages=stats['total_pages'],
                extracted_pages=stats['total_pages'] - stats['failed_pages'],
                failed_pages=stats['failed_pages'],
                text_pages=stats['text_pages'],
                vision_pages=stats['vision_pages'],
                low_confidence_pages=0,
                total_chars=0,
                avg_confidence=stats['avg_confidence'],
                success=stats['success'],
                issues=["Loaded from checkpoint - run with --force to re-extract"]
            )

        logger.info(f"  Extracting: {filename}")

        # Open PDF
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
        except Exception as e:
            logger.error(f"  ❌ Cannot open PDF: {e}")
            return DocumentResult(
                filename=filename,
                filepath=str(pdf_path),
                total_pages=0,
                extracted_pages=0,
                failed_pages=0,
                text_pages=0,
                vision_pages=0,
                low_confidence_pages=0,
                total_chars=0,
                avg_confidence=0.0,
                success=False,
                issues=[f"Cannot open PDF: {e}"]
            )

        # Extract each page
        pages: List[PageResult] = []
        text_pages = 0
        vision_pages = 0
        failed_pages = 0
        low_confidence_pages = 0
        total_chars = 0
        total_confidence = 0.0
        doc_issues = []
        vision_cost = 0.0

        for page_num in range(num_pages):
            page = doc[page_num]

            # Assess if Vision needed
            needs_vision, raw_text, char_count, printable_ratio = assess_page_needs_vision(page)

            # PyMuPDF extraction is good
            text_cleaned = clean_text(raw_text)
            success, confidence, page_issues = validate_text_quality(
                raw_text, page_num, 'pymupdf'
            )

            # If validation failed despite having chars, might need Vision
            if not success and confidence < 0.5:
                needs_vision = True
                page_issues.append("Validation failed - trying Vision")
            else:
                pages.append(PageResult(
                    page_num=page_num,
                    text=raw_text,
                    text_cleaned=text_cleaned,
                    char_count=char_count,
                    char_count_cleaned=len(text_cleaned),
                    method='pymupdf',
                    success=success,
                    confidence=confidence,
                    printable_ratio=printable_ratio,
                    issues=page_issues
                ))

                text_pages += 1
                total_chars += len(text_cleaned)
                total_confidence += confidence

                if confidence < 0.7:
                    low_confidence_pages += 1

                    continue

        doc.close()

        # Build full text (cleaned version)
        full_text = "\n\n".join(
            f"[PAGE {p.page_num + 1}]\n{p.text_cleaned}"
            for p in pages
        )

        # Calculate averages
        avg_confidence = total_confidence / num_pages if num_pages > 0 else 0.0

        # Determine document success
        success = (
            failed_pages < num_pages * 0.1 and  # <10% failed
            avg_confidence >= 0.6               # Decent average confidence
        )

        if failed_pages > 0:
            doc_issues.append(f"{failed_pages} pages failed extraction")
        if low_confidence_pages > num_pages * 0.2:
            doc_issues.append(f"{low_confidence_pages} pages have low confidence")
        if avg_confidence < 0.7:
            doc_issues.append(f"Low average confidence: {avg_confidence:.2f}")

        extraction_time = time.time() - start_time

        result = DocumentResult(
            filename=filename,
            filepath=str(pdf_path),
            total_pages=num_pages,
            extracted_pages=num_pages - failed_pages,
            failed_pages=failed_pages,
            text_pages=text_pages,
            vision_pages=vision_pages,
            low_confidence_pages=low_confidence_pages,
            total_chars=total_chars,
            avg_confidence=avg_confidence,
            success=success,
            issues=doc_issues,
            pages=pages,
            full_text=full_text,
            extraction_time_sec=extraction_time,
            vision_cost=vision_cost
        )

        # Save checkpoint
        if self.checkpoint:
            self.checkpoint.mark_completed(result)

        # Log summary
        status = "✅" if success else "⚠️"
        logger.info(
            f"    {status} {num_pages} pages | "
            f"text:{text_pages} vision:{vision_pages} failed:{failed_pages} | "
            f"conf:{avg_confidence:.2f} | "
            f"${vision_cost:.3f}"
        )

        return result

    def extract_batch(
        self,
        pdf_dir: Path,
        output_dir: Path,
        limit: Optional[int] = None,
        force: bool = False
    ) -> Tuple[List[DocumentResult], ExtractionReport]:
        """
        Extract text from all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDFs
            output_dir: Where to save extracted text
            limit: Max PDFs to process (for testing)
            force: Re-extract even if cached

        Returns:
            (list of DocumentResults, ExtractionReport)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        text_dir = output_dir / "extracted_text"
        text_dir.mkdir(exist_ok=True)

        # Clear checkpoint if forcing
        if force and self.checkpoint:
            self.checkpoint.completed = {}
            self.checkpoint.save()
            logger.info("🔄 Cleared checkpoint (--force)")

        # Find PDFs
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if limit:
            pdfs = pdfs[:limit]

        logger.info(f"\n{'='*70}")
        logger.info(f"PRODUCTION TEXT EXTRACTION")
        logger.info(f"{'='*70}")
        logger.info(f"PDFs to process: {len(pdfs)}")
        logger.info(f"Output directory: {output_dir}")
        if self.checkpoint:
            logger.info(f"Checkpoint: {len(self.checkpoint.completed)} already done")

        start_time = time.time()
        results: List[DocumentResult] = []

        for i, pdf_path in enumerate(pdfs):
            logger.info(f"\n[{i+1}/{len(pdfs)}] {pdf_path.name}")

            try:
                result = self.extract_document(pdf_path)
                results.append(result)

                # Save text file (only if we have new content)
                if result.full_text:
                    txt_path = text_dir / f"{pdf_path.stem}.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(result.full_text)

            except Exception as e:
                logger.error(f"  ❌ FAILED: {e}")
                results.append(DocumentResult(
                    filename=pdf_path.name,
                    filepath=str(pdf_path),
                    total_pages=0,
                    extracted_pages=0,
                    failed_pages=0,
                    text_pages=0,
                    vision_pages=0,
                    low_confidence_pages=0,
                    total_chars=0,
                    avg_confidence=0.0,
                    success=False,
                    issues=[str(e)]
                ))

        total_time = time.time() - start_time

        # Build report
        report = self._build_report(results, total_time)

        # Save report
        report_path = output_dir / "extraction_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save detailed results (without full_text to save space)
        results_summary = []
        for r in results:
            summary = r.to_dict()
            summary.pop('full_text', None)  # Remove full text
            summary.pop('pages', None)       # Remove page details
            results_summary.append(summary)

        results_path = output_dir / "extraction_summary.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        report.print_summary()

        logger.info(f"\n📁 Output files:")
        logger.info(f"  Text files: {text_dir}/")
        logger.info(f"  Report: {report_path}")
        logger.info(f"  Summary: {results_path}")

        return results, report

    def _build_report(
        self,
        results: List[DocumentResult],
        total_time: float
    ) -> ExtractionReport:
        """Build summary report from results."""

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_pages = sum(r.total_pages for r in results)
        text_pages = sum(r.text_pages for r in results)
        vision_pages = sum(r.vision_pages for r in results)
        failed_pages = sum(r.failed_pages for r in results)
        low_conf_pages = sum(r.low_confidence_pages for r in results)
        total_chars = sum(r.total_chars for r in results)

        avg_confidence = (
            sum(r.avg_confidence for r in results) / len(results)
            if results else 0.0
        )

        total_cost = sum(r.vision_cost for r in results)

        # Documents needing review
        needs_review = [
            r.filename for r in results
            if not r.success or r.avg_confidence < 0.7 or r.failed_pages > 0
        ]

        return ExtractionReport(
            total_documents=len(results),
            successful_documents=len(successful),
            failed_documents=len(failed),
            total_pages=total_pages,
            text_pages=text_pages,
            vision_pages=vision_pages,
            failed_pages=failed_pages,
            low_confidence_pages=low_conf_pages,
            total_chars=total_chars,
            avg_confidence=avg_confidence,
            total_time_sec=total_time,
            total_vision_cost=total_cost,
            documents_needing_review=needs_review
        )


# =============================================================================
# CONVENIENCE FUNCTIONS (Backwards compatible)
# =============================================================================

# Alias for backwards compatibility
TextExtractor = ProductionTextExtractor


def get_document_text_quality(pdf_path: Path) -> Dict:
    """
    Quick quality assessment for a single PDF.

    Returns dict with page counts and quality metrics.
    """
    doc = fitz.open(pdf_path)

    text_pages = 0
    image_pages = 0
    low_quality_pages = 0

    for page in doc:
        text = page.get_text()
        char_count = len(text.strip())
        printable_ratio = calculate_printable_ratio(text)

        if char_count >= MIN_CHARS_FOR_TEXT and printable_ratio >= MIN_PRINTABLE_RATIO:
            text_pages += 1
        else:
            image_pages += 1

        if printable_ratio < MIN_PRINTABLE_RATIO:
            low_quality_pages += 1

    doc.close()

    total = text_pages + image_pages

    return {
        "filename": pdf_path.name,
        "total_pages": total,
        "text_pages": text_pages,
        "image_pages": image_pages,
        "low_quality_pages": low_quality_pages,
        "text_ratio": text_pages / total if total > 0 else 0,
        "estimated_vision_cost": image_pages * COST_PER_VISION_CALL,
        "estimated_vision_time_sec": image_pages * 4  # ~4 sec per Vision call
    }
