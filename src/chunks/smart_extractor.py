"""
Smart Extractor - Intelligent SGIA Field Extraction

Architecture:
1. Load metadata from enriched CSV (skip known fields)
2. Detect PDF type (text vs scanned)
3. Extract only key sections (skip boilerplate Articles 1-10)
4. Use Haiku for text (cheap), Sonnet for vision (accurate)
5. Auto-sum security components if total not stated
6. Validate and compute derived fields

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import json
import logging
import base64
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field

import fitz  # PyMuPDF
import anthropic

from .metadata import MetadataRegistry, ProjectMetadata
from .prompts_v2 import (
    EXTRACTION_PROMPT_V2,
    RETRY_SECURITY_PROMPT_V2,
    PARENT_COMPANY_PROMPT,
)
from .prompts import (
    VISION_EXTRACTION_PROMPT,
    RETRY_TSP_PROMPT,
    RETRY_PARENT_PROMPT,
)
from .parent_extractor import ParentCompanyExtractor
from .checkpoint import CheckpointManager
from .validation import validate_extraction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model selection
HAIKU_MODEL = "claude-3-5-haiku-20241022"   # Fast, cheap for text extraction
SONNET_MODEL = "claude-sonnet-4-20250514"   # Accurate for vision (scanned PDFs)

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 2

# Token/page limits
MAX_TEXT_CHARS = 100000  # ~25k tokens
MAX_VISION_PAGES = 10    # Max pages to send as images

# Quality thresholds
LOW_TEXT_QUALITY = 0.5   # Below this, use vision (raised from 0.3)
HIGH_CONFIDENCE = 0.7    # Above this, skip retry


# =============================================================================
# TSP NORMALIZATION
# =============================================================================

TSP_NORMALIZE = {
    'ETT': ['electric transmission texas', 'ett'],
    'ONCOR': ['oncor'],
    'CENTERPOINT': ['centerpoint', 'center point', 'cnp'],
    'LCRA': ['lcra', 'lower colorado river authority'],
    'AEP': ['aep texas', 'aep '],
    'SHARYLAND': ['sharyland'],
    'LONE STAR': ['lone star transmission'],
    'CROSS TEXAS': ['cross texas'],
    'TNMP': ['texas-new mexico', 'tnmp'],
    'BRAZOS': ['brazos electric'],
    'CPS': ['cps energy'],
}


def normalize_tsp(tsp_name: str) -> str:
    """Normalize TSP name to standard abbreviation."""
    if not tsp_name:
        return ""
    tsp_lower = tsp_name.lower()
    for abbrev, keywords in TSP_NORMALIZE.items():
        if any(kw in tsp_lower for kw in keywords):
            return abbrev
    return tsp_name


# =============================================================================
# EXTRACTION RESULT
# =============================================================================

@dataclass
class ExtractionResult:
    """Complete extraction result for one PDF."""

    # File info
    filename: str
    filepath: str
    extraction_timestamp: str

    # From metadata (pre-loaded)
    inr: str = ""
    item_number: str = ""
    project_name: str = ""
    developer_spv: str = ""
    parent_company: str = ""          # Final parent (best source wins)
    capacity_mw: float = 0.0
    county: str = ""
    zone: str = ""
    fuel_type: str = ""
    technology: str = ""

    # Extracted from PDF (competitive intelligence)
    tsp_name: str = ""
    tsp_normalized: str = ""
    security_design_usd: float = 0.0
    security_construction_usd: float = 0.0
    security_total_usd: float = 0.0      # Auto-summed if not explicit
    security_per_kw: float = 0.0         # Computed
    is_amended: bool = False
    amendment_type: str = ""
    commercial_operation_date: str = ""

    # CEO-GRADE: Hidden Parent Intelligence (V4+)
    ir_submitter: str = ""               # Company that filed IR (from recitals) - MOST RELIABLE
    parent_company_inferred: str = ""    # LLM's guess at true parent
    parent_evidence: str = ""            # Quote showing how we found parent
    email_domains_found: List[str] = field(default_factory=list)  # Non-TSP domains

    # Equipment (secondary priority)
    inverter_manufacturer: str = ""
    inverter_model: str = ""
    inverter_quantity: int = 0
    turbine_manufacturer: str = ""
    turbine_model: str = ""
    turbine_quantity: int = 0

    # Quality metrics
    text_quality: float = 0.0
    used_vision: bool = False
    extraction_confidence: float = 0.0
    fields_extracted: int = 0
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def compute_derived(self):
        """Calculate derived fields after extraction."""

        # AUTO-SUMMATION: If total is 0 but components exist, sum them
        if self.security_total_usd == 0:
            design = self.security_design_usd or 0
            construction = self.security_construction_usd or 0
            if design > 0 or construction > 0:
                self.security_total_usd = design + construction

        # Security per kW
        if self.capacity_mw > 0 and self.security_total_usd > 0:
            self.security_per_kw = round(
                self.security_total_usd / (self.capacity_mw * 1000), 2
            )

        # TSP normalized
        if self.tsp_name and not self.tsp_normalized:
            self.tsp_normalized = normalize_tsp(self.tsp_name)

        # Count extracted fields
        self._count_fields()

    def _count_fields(self):
        """Count non-empty critical fields."""
        # V5.2: Removed COD - already in Gold Master CSV
        critical = ['security_total_usd', 'tsp_name', 'parent_company', 'is_amended']
        count = 0
        for f in critical:
            val = getattr(self, f, None)
            if val and val != 0 and val != "":
                count += 1
        self.fields_extracted = count

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return asdict(self)


# =============================================================================
# SECTION DETECTOR
# =============================================================================

class SectionDetector:
    """Detect key sections in SGIA PDFs."""

    def __init__(self):
        self.sections = {}

    def detect_sections(self, pdf_path: Path) -> Dict[str, str]:
        """
        Extract text from key sections only.

        Returns:
            Dict with section name -> text content
        """
        sections = {
            'cover_letter': '',
            'exhibit_b': '',
            'exhibit_c': '',
            'exhibit_de': '',  # D and E combined
        }

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)

            # Collect all text with page markers
            all_text = []
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                all_text.append((page_num, text))

            doc.close()

            # Extract sections
            full_text = "\n".join(t for _, t in all_text)

            # Cover letter: First 2-3 pages
            sections['cover_letter'] = "\n".join(t for p, t in all_text if p < 3)[:15000]

            # Find exhibits by searching text
            full_lower = full_text.lower()

            # Exhibit B (Timeline)
            if 'exhibit b' in full_lower:
                start = full_lower.find('exhibit b')
                sections['exhibit_b'] = full_text[start:start+10000]

            # Exhibit C (Equipment)
            if 'exhibit c' in full_lower:
                start = full_lower.find('exhibit c')
                sections['exhibit_c'] = full_text[start:start+15000]

            # Exhibit D/E (Security & Notices)
            for exhibit in ['exhibit d', 'exhibit e']:
                if exhibit in full_lower:
                    start = full_lower.find(exhibit)
                    sections['exhibit_de'] += full_text[start:start+10000] + "\n"

            # If no exhibits found, use last 30% of document
            if not sections['exhibit_de']:
                last_third_start = len(full_text) * 2 // 3
                sections['exhibit_de'] = full_text[last_third_start:][:20000]

        except Exception as e:
            logger.error(f"Error detecting sections: {e}")

        return sections

    def get_text_quality(self, pdf_path: Path) -> float:
        """
        Estimate text quality (0-1).

        Low quality = likely scanned, needs vision.
        """
        try:
            doc = fitz.open(pdf_path)

            total_chars = 0
            total_pages = len(doc)

            for page in doc:
                text = page.get_text()
                total_chars += len(text)

            doc.close()

            # Expected: ~3000 chars per page for text-based PDFs
            expected_chars = total_pages * 3000
            quality = min(1.0, total_chars / expected_chars) if expected_chars > 0 else 0

            return quality

        except Exception as e:
            logger.error(f"Error checking quality: {e}")
            return 0.0

    def get_pages_for_vision(self, pdf_path: Path) -> List[int]:
        """Get key page numbers for vision extraction."""
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()

            pages = []

            # First 3 pages (cover letter)
            pages.extend([0, 1, 2])

            # Last 10 pages (exhibits)
            if num_pages > 15:
                pages.extend(range(num_pages - 10, num_pages))
            else:
                pages.extend(range(3, num_pages))

            # Dedupe and limit
            pages = sorted(set(pages))[:MAX_VISION_PAGES]

            return pages

        except Exception as e:
            logger.error(f"Error getting pages: {e}")
            return [0, 1]


# =============================================================================
# SMART EXTRACTOR
# =============================================================================

class SmartExtractor:
    """
    Intelligent SGIA field extractor.

    - Uses metadata from enriched CSV
    - Haiku for text-based PDFs
    - Sonnet + vision for scanned PDFs
    - Auto-sums security components
    - Retries for missing critical fields
    """

    def __init__(
        self,
        api_key: str,
        metadata_registry: Optional[MetadataRegistry] = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.section_detector = SectionDetector()
        self.metadata = metadata_registry

        # Stats
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'used_haiku': 0,
            'used_sonnet': 0,
            'retries': 0,
        }

    def _call_llm(
        self,
        prompt: str,
        text: str = "",
        model: str = HAIKU_MODEL,
        images: List[str] = None,
    ) -> Optional[Dict]:
        """
        Call Anthropic API with text or vision.
        """
        try:
            content = []

            if images:
                # Vision mode
                for img_b64 in images[:MAX_VISION_PAGES]:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        }
                    })
                content.append({"type": "text", "text": prompt})
            else:
                # Text mode - truncate if needed
                if len(text) > MAX_TEXT_CHARS:
                    text = text[:MAX_TEXT_CHARS]
                content.append({"type": "text", "text": prompt + text})

            response = self.client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.0,  # Zero for deterministic JSON
                system="You are a data extraction assistant. Return ONLY valid JSON. No markdown, no explanation, no preamble. Start with { and end with }.",
                messages=[{"role": "user", "content": content}]
            )

            response_text = response.content[0].text.strip()

            # ROBUST JSON CLEANING
            # Remove markdown code blocks anywhere in response
            if "```" in response_text:
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            # Try to find JSON object in response
            if not response_text.startswith("{"):
                # Look for JSON object in the text
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    response_text = json_match.group(0)
                else:
                    logger.warning(f"No JSON found in response: {response_text[:200]}...")
                    return None

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {response_text[:500] if 'response_text' in locals() else 'N/A'}")
            return None
        except anthropic.RateLimitError:
            logger.warning("Rate limited, waiting 30s...")
            time.sleep(30)
            return None
        except Exception as e:
            logger.error(f"API error: {e}")
            return None

    def _convert_pages_to_images(self, pdf_path: Path, pages: List[int]) -> List[str]:
        """Convert PDF pages to base64 images."""
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in pages:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    images.append(base64.b64encode(img_bytes).decode('utf-8'))
            doc.close()
        except Exception as e:
            logger.error(f"Error converting pages: {e}")
        return images

    def _extract_with_text(self, sections: Dict[str, str]) -> Optional[Dict]:
        """Extract using text mode (Haiku)."""

        # Combine relevant sections
        combined = ""
        for name in ['cover_letter', 'exhibit_de', 'exhibit_b', 'exhibit_c']:
            if sections.get(name):
                combined += f"\n=== {name.upper()} ===\n{sections[name]}\n"

        if len(combined) < 500:
            logger.warning("Very little text extracted")
            return None

        result = self._call_llm(EXTRACTION_PROMPT_V2, combined, HAIKU_MODEL)
        if result:
            self.stats['used_haiku'] += 1
        return result

    def _extract_with_vision(self, pdf_path: Path) -> Optional[Dict]:
        """Extract using vision mode (Sonnet)."""

        pages = self.section_detector.get_pages_for_vision(pdf_path)
        images = self._convert_pages_to_images(pdf_path, pages)

        if not images:
            logger.warning("Could not convert pages to images")
            return None

        result = self._call_llm(VISION_EXTRACTION_PROMPT, "", SONNET_MODEL, images)
        if result:
            self.stats['used_sonnet'] += 1
        return result

    def _retry_missing_fields(self, sections: Dict[str, str], extracted: Dict) -> Dict:
        """Retry extraction for missing critical fields."""

        updated = extracted.copy()
        combined_text = "\n".join(sections.values())

        # Retry security if missing (use V2 prompt with few-shot examples)
        if not extracted.get('security_design_usd') and not extracted.get('security_construction_usd'):
            logger.info("  Retrying security extraction...")
            result = self._call_llm(RETRY_SECURITY_PROMPT_V2, combined_text, HAIKU_MODEL)
            if result:
                if result.get('security_design_usd'):
                    updated['security_design_usd'] = result['security_design_usd']
                if result.get('security_construction_usd'):
                    updated['security_construction_usd'] = result['security_construction_usd']
                if result.get('security_total_usd'):
                    updated['security_total_usd'] = result['security_total_usd']
                self.stats['retries'] += 1

        # Retry TSP if missing
        if not extracted.get('tsp_name'):
            logger.info("  Retrying TSP extraction...")
            result = self._call_llm(RETRY_TSP_PROMPT, combined_text, HAIKU_MODEL)
            if result and result.get('tsp_name'):
                updated['tsp_name'] = result['tsp_name']
                self.stats['retries'] += 1

        # Retry Parent if missing (CEO-grade)
        if not extracted.get('parent_company_inferred') and not extracted.get('ir_submitter'):
            logger.info("  Retrying parent company extraction...")
            result = self._call_llm(RETRY_PARENT_PROMPT, combined_text, HAIKU_MODEL)
            if result:
                if result.get('ir_submitter'):
                    updated['ir_submitter'] = result['ir_submitter']
                if result.get('parent_company_inferred'):
                    updated['parent_company_inferred'] = result['parent_company_inferred']
                if result.get('email_domains_found'):
                    updated['email_domains_found'] = result['email_domains_found']
                if result.get('evidence'):
                    updated['parent_evidence'] = result['evidence']
                self.stats['retries'] += 1

        return updated

    def _retry_vision_with_more_pages(self, pdf_path: Path) -> Optional[Dict]:
        """
        V5.2: Retry vision extraction with MORE pages.

        For mixed PDFs where security table might be on a page we didn't capture.
        Sends up to 20 pages instead of 10.
        """
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()

            # Get more pages - focus on exhibits (where security usually is)
            pages = []

            # First 5 pages (cover letter + early exhibits)
            pages.extend(range(min(5, num_pages)))

            # Last 15 pages (all exhibits)
            if num_pages > 20:
                pages.extend(range(num_pages - 15, num_pages))
            else:
                pages.extend(range(5, num_pages))

            # Dedupe and cap at 20
            pages = sorted(set(pages))[:20]

            logger.info(f"    Retry vision with {len(pages)} pages (was {MAX_VISION_PAGES})")

            images = self._convert_pages_to_images(pdf_path, pages)
            if not images:
                return None

            # Use focused security prompt
            SECURITY_RETRY_VISION = """Look CAREFULLY at these document images for SECURITY AMOUNTS.

The security table is usually in Exhibit D or E. It looks like:
- "Maximum Stated Amount: $X,XXX,XXX"
- "Initial amount for Design: $X"
- "Additional amount for Construction: $Y"

FIND THE DOLLAR AMOUNTS and return:
{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0
}

Convert "Three Million Five Hundred Thousand Dollars" = 3500000
"""

            result = self._call_llm(SECURITY_RETRY_VISION, "", SONNET_MODEL, images[:20])
            if result:
                self.stats['retries'] += 1
                logger.info(f"    Retry found: D=${result.get('security_design_usd', 0):,}, C=${result.get('security_construction_usd', 0):,}")
            return result

        except Exception as e:
            logger.error(f"Vision retry error: {e}")
            return None

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """
        Extract fields from a single PDF.
        """
        self.stats['total'] += 1
        filename = pdf_path.name

        logger.info(f"Extracting: {filename}")

        # Get metadata from registry
        meta = None
        if self.metadata:
            meta = self.metadata.get_by_filename(filename)
            if meta:
                logger.info(f"  Metadata: INR={meta.inr}, Project={meta.project_name}")

        # Initialize result
        result = ExtractionResult(
            filename=filename,
            filepath=str(pdf_path),
            extraction_timestamp=datetime.now().isoformat(),
        )

        # Copy metadata fields
        if meta:
            result.inr = meta.inr
            result.item_number = meta.item_number
            result.project_name = meta.project_name
            result.developer_spv = meta.developer_spv
            result.parent_company = meta.parent_company  # From intelligence layer
            result.capacity_mw = meta.capacity_mw
            result.county = meta.county
            result.zone = meta.zone
            result.fuel_type = meta.fuel_type
            result.technology = meta.technology
            result.is_amended = meta.is_amended  # Pre-detected

        # Check text quality
        text_quality = self.section_detector.get_text_quality(pdf_path)
        result.text_quality = text_quality

        # Extract sections
        sections = self.section_detector.detect_sections(pdf_path)

        # Choose extraction method
        extracted = None
        if text_quality < LOW_TEXT_QUALITY:
            logger.info(f"  Using VISION mode (quality={text_quality:.2f})")
            extracted = self._extract_with_vision(pdf_path)
            result.used_vision = True
        else:
            logger.info(f"  Using TEXT mode (quality={text_quality:.2f})")
            extracted = self._extract_with_text(sections)

        # Handle extraction failure
        if not extracted:
            logger.warning(f"  ✗ Extraction failed")
            result.needs_review = True
            result.review_reasons.append("LLM extraction failed")
            self.stats['failed'] += 1
            return result

        # Apply extracted values
        result.tsp_name = extracted.get('tsp_name', '')
        result.security_design_usd = float(extracted.get('security_design_usd', 0) or 0)
        result.security_construction_usd = float(extracted.get('security_construction_usd', 0) or 0)
        result.security_total_usd = float(extracted.get('security_total_usd', 0) or 0)
        result.amendment_type = extracted.get('amendment_type', '')
        result.commercial_operation_date = extracted.get('commercial_operation_date', '')
        result.extraction_confidence = float(extracted.get('extraction_confidence', 0.5) or 0.5)

        # Equipment
        result.inverter_manufacturer = extracted.get('inverter_manufacturer', '')
        result.inverter_model = extracted.get('inverter_model', '')
        result.inverter_quantity = int(extracted.get('inverter_quantity', 0) or 0)
        result.turbine_manufacturer = extracted.get('turbine_manufacturer', '')
        result.turbine_model = extracted.get('turbine_model', '')
        result.turbine_quantity = int(extracted.get('turbine_quantity', 0) or 0)

        # =====================================================================
        # CEO-GRADE: Hidden Parent Intelligence (V5 - Hybrid Regex + LLM)
        # =====================================================================
        # Priority: 1) Regex IR Submitter (free), 2) LLM extraction, 3) Metadata

        # Step 1: Try regex-based extraction (FREE - no LLM cost)
        full_text = "\n".join(sections.values()) if sections else ""
        parent_extractor = ParentCompanyExtractor()
        regex_result = parent_extractor.extract(full_text)

        if regex_result.ir_submitter_raw:
            result.ir_submitter = regex_result.ir_submitter_raw
            logger.info(f"    IR Submitter (regex): {result.ir_submitter}")

        if regex_result.email_domains:
            result.email_domains_found = regex_result.email_domains
            logger.info(f"    Email domains (regex): {result.email_domains_found}")

        if regex_result.parent_company and regex_result.parent_company != 'UNKNOWN':
            result.parent_company_inferred = regex_result.parent_company
            result.parent_evidence = f"Source: {regex_result.source}, Confidence: {regex_result.confidence}"

        # Step 2: Also capture LLM extraction if available
        if extracted.get('ir_submitter') and not result.ir_submitter:
            result.ir_submitter = extracted['ir_submitter']
        if extracted.get('parent_company_inferred'):
            result.parent_company_inferred = extracted['parent_company_inferred']
        if extracted.get('parent_evidence'):
            result.parent_evidence = extracted['parent_evidence']
        if extracted.get('email_domains_found') and not result.email_domains_found:
            result.email_domains_found = extracted['email_domains_found']

        # Step 3: Smart Parent Resolution (best source wins)
        # Priority: 1) Regex IR submitter normalized, 2) LLM inferred, 3) Metadata mapping
        if result.parent_company_inferred and result.parent_company_inferred.upper() not in ['', 'NULL', 'NONE', 'UNKNOWN']:
            final_parent = result.parent_company_inferred.upper()
            logger.info(f"    Parent from PDF: {final_parent}")
        elif result.ir_submitter:
            final_parent = result.ir_submitter
            logger.info(f"    Parent from IR submitter: {final_parent}")
        else:
            final_parent = result.parent_company  # Keep metadata value

        if final_parent and final_parent != 'UNKNOWN':
            result.parent_company = final_parent

        # Override is_amended if PDF says so
        if extracted.get('is_amended'):
            result.is_amended = True

        # Retry if missing critical fields
        has_security = result.security_design_usd > 0 or result.security_construction_usd > 0
        if not has_security or not result.tsp_name:
            logger.info("  Missing critical fields, retrying...")
            retry_result = self._retry_missing_fields(sections, extracted)

            if retry_result.get('security_design_usd') and not result.security_design_usd:
                result.security_design_usd = float(retry_result['security_design_usd'])
            if retry_result.get('security_construction_usd') and not result.security_construction_usd:
                result.security_construction_usd = float(retry_result['security_construction_usd'])
            if retry_result.get('security_total_usd') and not result.security_total_usd:
                result.security_total_usd = float(retry_result['security_total_usd'])
            if retry_result.get('tsp_name') and not result.tsp_name:
                result.tsp_name = retry_result['tsp_name']

        # Compute derived fields (includes auto-summation!)
        result.compute_derived()

        # =====================================================================
        # V5.2 FIX 1: COLOCATION DETECTION
        # =====================================================================
        # Some projects share existing POI and don't require security
        full_text_lower = full_text.lower() if full_text else ""
        colocation_phrases = [
            "security instrument will not be required",
            "security will not be required",
            "no security instrument",
            "utilizing an existing point of interconnection",
            "existing poi",
        ]
        is_colocation = any(phrase in full_text_lower for phrase in colocation_phrases)

        if is_colocation and result.security_total_usd == 0:
            # This is a valid zero - not a failure
            result.needs_review = False
            result.review_reasons = ["Colocation - no security required (valid)"]
            logger.info("  ℹ️  Colocation detected - no security required")

        # =====================================================================
        # V5.2 FIX 2: MIXED PDF RETRY
        # =====================================================================
        # If vision mode failed to find security, retry with more pages
        if result.security_total_usd == 0 and result.used_vision and not is_colocation:
            logger.info("  🔄 Vision missed security - retrying with more pages...")
            retry_extracted = self._retry_vision_with_more_pages(pdf_path)
            if retry_extracted:
                if retry_extracted.get('security_design_usd'):
                    result.security_design_usd = float(retry_extracted['security_design_usd'])
                if retry_extracted.get('security_construction_usd'):
                    result.security_construction_usd = float(retry_extracted['security_construction_usd'])
                if retry_extracted.get('security_total_usd'):
                    result.security_total_usd = float(retry_extracted['security_total_usd'])
                # Recompute derived
                result.compute_derived()

        # Check if needs review (skip if already handled by colocation)
        if not is_colocation:
            if result.security_total_usd == 0:
                result.needs_review = True
                result.review_reasons.append("Security amount not found")
            if result.extraction_confidence < 0.5:
                result.needs_review = True
                result.review_reasons.append("Low extraction confidence")

        # =====================================================================
        # VALIDATION: Range checks and anomaly detection (V5)
        # =====================================================================
        try:
            validation = validate_extraction(result.to_dict())
            if not validation.is_valid:
                result.needs_review = True
                for issue in validation.issues:
                    if issue not in result.review_reasons:
                        result.review_reasons.append(issue)
                logger.warning(f"  ⚠️  Validation issues: {validation.issues}")
        except Exception as e:
            logger.debug(f"Validation error (non-fatal): {e}")

        # Rate limit
        time.sleep(REQUEST_DELAY)

        self.stats['successful'] += 1
        logger.info(f"  ✓ Security: ${result.security_total_usd:,.0f} (D:${result.security_design_usd:,.0f} + C:${result.security_construction_usd:,.0f})")
        logger.info(f"    TSP: {result.tsp_normalized}, Parent: {result.parent_company}, Zone: {result.zone}")

        return result

    def extract_batch(
        self,
        pdf_dir: Path,
        output_dir: Path,
        limit: Optional[int] = None,
    ) -> List[ExtractionResult]:
        """
        Extract from all PDFs in directory.

        Features:
        - Checkpoint/resume: If extraction crashes, resume where you left off
        - Progress tracking: See how many documents processed
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find PDFs
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if limit:
            pdfs = pdfs[:limit]

        # =====================================================================
        # CHECKPOINT: Resume capability (V5)
        # =====================================================================
        checkpoint = CheckpointManager(str(output_dir / "checkpoint.json"))
        skip_files = set()

        if checkpoint.has_checkpoint():
            state = checkpoint.load()
            skip_files = set(state.processed_files)
            logger.info(f"📍 RESUMING: {len(skip_files)} already processed, skipping...")
        else:
            checkpoint.state.total_documents = len(pdfs)
            checkpoint.state.start_time = datetime.now().isoformat()

        logger.info(f"Processing {len(pdfs)} PDFs ({len(pdfs) - len(skip_files)} remaining)")

        results = []
        for i, pdf_path in enumerate(pdfs, 1):
            # Skip already processed files
            if pdf_path.name in skip_files:
                logger.info(f"[{i}/{len(pdfs)}] ⏭️  Skipping (already processed): {pdf_path.name}")
                continue

            logger.info(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")

            try:
                result = self.extract(pdf_path)
                results.append(result)

                # Update checkpoint
                checkpoint.mark_processed(pdf_path.name, success=not result.needs_review)
                checkpoint.save()

            except Exception as e:
                logger.error(f"Error: {e}")
                results.append(ExtractionResult(
                    filename=pdf_path.name,
                    filepath=str(pdf_path),
                    extraction_timestamp=datetime.now().isoformat(),
                    needs_review=True,
                    review_reasons=[f"Error: {str(e)}"],
                ))
                # Still checkpoint failures
                checkpoint.mark_processed(pdf_path.name, success=False)
                checkpoint.save()

        # Save results
        results_file = output_dir / "extraction_results.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"\n✓ Saved: {results_file}")

        # Save needs review
        needs_review = [r for r in results if r.needs_review]
        review_file = output_dir / "needs_review.json"
        with open(review_file, 'w') as f:
            json.dump([r.to_dict() for r in needs_review], f, indent=2)

        # Print summary
        self._print_summary(results)

        # Clear checkpoint on successful completion
        if len(results) == len(pdfs) - len(skip_files):
            checkpoint.clear()
            logger.info("✓ Checkpoint cleared (extraction complete)")

        return results

    def _print_summary(self, results: List[ExtractionResult]):
        """Print extraction summary."""
        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {self.stats['total']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Used Haiku: {self.stats['used_haiku']}")
        logger.info(f"Used Sonnet (vision): {self.stats['used_sonnet']}")
        logger.info(f"Retries: {self.stats['retries']}")

        # Field coverage
        n = len(results)
        security = sum(1 for r in results if r.security_total_usd > 0)
        tsp = sum(1 for r in results if r.tsp_name)
        parent = sum(1 for r in results if r.parent_company and r.parent_company != 'UNKNOWN')

        logger.info(f"\nField Coverage:")
        logger.info(f"  Security: {security}/{n} ({100*security/n:.0f}%)")
        logger.info(f"  TSP: {tsp}/{n} ({100*tsp/n:.0f}%)")
        logger.info(f"  Parent: {parent}/{n} ({100*parent/n:.0f}%)")

        # Needs review
        review = sum(1 for r in results if r.needs_review)
        logger.info(f"  Needs Review: {review}/{n} ({100*review/n:.0f}%)")

    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return self.stats.copy()
