"""
Parent Company Extractor - Regex + LLM Hybrid

Uses the IR Submitter pattern discovered from SGIA analysis:
- PRIMARY: "interconnection request #XXINRXXXX to ERCOT from [COMPANY]"
- SECONDARY: "c/o [COMPANY]" in notice addresses
- TERTIARY: Email domains from non-utility contacts

This is the "hidden intelligence" that no public database has.

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# PARENT COMPANY NORMALIZATION
# =============================================================================

PARENT_NORMALIZE = {
    # Exact matches (case-insensitive)
    'SAMSUNG C&T': ['samsung c&t', 'samsung c & t', 'samsung ct'],
    'SAMSUNG': ['samsung'],
    'KOSPO': ['kospo', 'korea south-east power', 'korea southeast power'],
    'KEPCO': ['kepco', 'korea electric power'],
    'NEXTERA': ['nextera', 'fpl', 'florida power & light', 'nee'],
    'RWE': ['rwe renewables', 'rwe solar', 'rwe clean'],
    'INVENERGY': ['invenergy'],
    'EDF': ['edf renewables', 'edf re'],
    'ENEL': ['enel green power', 'enel'],
    'AES': ['aes corporation', 'aes '],
    'ENGIE': ['engie'],
    'ORSTED': ['orsted', 'ørsted'],
    'AVANGRID': ['avangrid'],
    'PATTERN': ['pattern energy'],
    'EDP': ['edp renewables'],
    'TERRA-GEN': ['terra-gen', 'terragen'],
    'CANADIAN SOLAR': ['canadian solar', 'recurrent energy'],
    'LIGHTSOURCE BP': ['lightsource', 'lightsource bp'],
    'CIP': ['copenhagen infrastructure'],
    'SAVION': ['savion'],
    'CLEARWAY': ['clearway'],
    'APEX': ['apex clean energy'],
    'HECATE': ['hecate energy'],
    'LEEWARD': ['leeward renewable'],
    'LONGROAD': ['longroad energy'],
    'CYPRESS CREEK': ['cypress creek'],
    '174 POWER': ['174 power global'],
    'PLUS POWER': ['plus power'],
    'KEY CAPTURE': ['key capture energy'],
    'BROAD REACH': ['broad reach power'],
    'JUPITER': ['jupiter power'],
    'ABLE GRID': ['able grid'],
    'VISTRA': ['vistra'],
    'NRG': ['nrg energy', 'nrg '],
    'DUKE': ['duke energy', 'sweetwater wind'],
    'SOUTHERN': ['southern company', 'southern power'],
    'CPV': ['cpv', 'competitive power'],
    'GRANSOLAR': ['gransolar'],
    'MISAE': ['misae', 'excel advantage'],
    'INTERSECT': ['intersect power'],
    '8MINUTE': ['8minute', '8 minute solar'],
    'SILICON RANCH': ['silicon ranch'],
    'ORIGIS': ['origis energy'],
    'SOL SYSTEMS': ['sol systems'],
    'TRI GLOBAL': ['tri global', 'tri-global'],
    'SUNCHASE': ['sunchase', 'sunchasepower', 'sunchase power'],
    'TESLA': ['tesla', 'giga texas'],
}


def normalize_parent(raw_name: str) -> str:
    """
    Normalize raw company name to standard parent.
    
    Args:
        raw_name: Raw company name from document
    
    Returns:
        Normalized parent company name
    """
    if not raw_name:
        return ''
    
    raw_lower = raw_name.lower().strip()
    
    for parent, keywords in PARENT_NORMALIZE.items():
        if any(kw in raw_lower for kw in keywords):
            return parent
    
    # If no match, return cleaned version of original
    # Remove common suffixes
    cleaned = raw_name.strip()
    for suffix in [', LLC', ' LLC', ', Inc.', ' Inc.', ', LP', ' LP', ', L.P.', ' L.P.']:
        cleaned = cleaned.replace(suffix, '')
    
    return cleaned.strip()


# =============================================================================
# EXTRACTION RESULT
# =============================================================================

@dataclass
class ParentExtractionResult:
    """Result of parent company extraction."""
    
    # Primary source: IR submitter
    ir_submitter_raw: str = ""
    ir_submitter_inr: str = ""
    
    # Secondary source: c/o address
    co_company_raw: str = ""
    
    # Tertiary source: email domains
    email_domains: List[str] = None
    
    # Final result
    parent_company: str = ""
    confidence: float = 0.0
    source: str = ""  # 'ir_submitter', 'co_address', 'email', 'unknown'
    
    def __post_init__(self):
        if self.email_domains is None:
            self.email_domains = []


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class ParentCompanyExtractor:
    """
    Extract parent company from SGIA text using multiple patterns.
    
    Priority:
    1. IR Submitter (recitals) - 100% reliable
    2. c/o Address (Exhibit D/E) - 80% reliable
    3. Email domains - 60% reliable
    """
    
    # Regex patterns
    IR_SUBMITTER_PATTERN = re.compile(
        r'interconnection\s+request\s+#?(\d{2}INR\d+)\s+to\s+ERCOT\s+from\s+([^\.]+)',
        re.IGNORECASE
    )
    
    CO_ADDRESS_PATTERN = re.compile(
        r'c/o\s+([A-Z][A-Za-z\s&\.,]+(?:LLC|Inc|LP|Corporation|Company)?)',
        re.IGNORECASE
    )
    
    EMAIL_PATTERN = re.compile(
        r'[\w\.-]+@([\w\.-]+\.\w+)',
        re.IGNORECASE
    )
    
    # Utility domains to exclude
    UTILITY_DOMAINS = {
        'aep.com', 'oncor.com', 'centerpointenergy.com', 'txu.com',
        'ettexas.com', 'lcra.org', 'sharyland.com', 'cps.com',
        'ercot.com', 'puc.texas.gov', 'gmail.com', 'outlook.com',
    }
    
    def extract(self, text: str) -> ParentExtractionResult:
        """
        Extract parent company from SGIA text.
        
        Args:
            text: Full extracted text from PDF
        
        Returns:
            ParentExtractionResult with best guess
        """
        result = ParentExtractionResult()
        
        # 1. Try IR Submitter pattern (BEST)
        ir_match = self.IR_SUBMITTER_PATTERN.search(text)
        if ir_match:
            result.ir_submitter_inr = ir_match.group(1)
            result.ir_submitter_raw = ir_match.group(2).strip()
            
            # Clean up trailing punctuation/words
            raw = result.ir_submitter_raw
            raw = re.sub(r'\s*\.\s*$', '', raw)  # Remove trailing period
            raw = re.sub(r'\s+(Transmission|Generator|Plant).*$', '', raw, flags=re.IGNORECASE)
            result.ir_submitter_raw = raw.strip()
            
            parent = normalize_parent(result.ir_submitter_raw)
            if parent:
                result.parent_company = parent
                result.confidence = 0.95
                result.source = 'ir_submitter'
                logger.info(f"Found IR Submitter: {result.ir_submitter_raw} → {parent}")
                return result
        
        # 2. Try c/o Address pattern
        co_matches = self.CO_ADDRESS_PATTERN.findall(text)
        if co_matches:
            # Filter out utility service companies
            for co_company in co_matches:
                co_lower = co_company.lower()
                # Skip if it's a utility service company
                if any(x in co_lower for x in ['electric power service', 'transmission', 'utility']):
                    continue
                
                result.co_company_raw = co_company.strip()
                parent = normalize_parent(result.co_company_raw)
                if parent:
                    result.parent_company = parent
                    result.confidence = 0.75
                    result.source = 'co_address'
                    logger.info(f"Found c/o: {result.co_company_raw} → {parent}")
                    return result
        
        # 3. Try email domains
        emails = self.EMAIL_PATTERN.findall(text)
        unique_domains = set()
        for domain in emails:
            domain_lower = domain.lower()
            if domain_lower not in self.UTILITY_DOMAINS:
                unique_domains.add(domain_lower)
        
        result.email_domains = list(unique_domains)
        
        # Try to match email domains to known parents
        for domain in unique_domains:
            domain_base = domain.split('.')[0]  # e.g., 'kospo' from 'kospo.co.kr'
            parent = normalize_parent(domain_base)
            if parent and parent != domain_base:  # Only if we found a match
                result.parent_company = parent
                result.confidence = 0.60
                result.source = 'email'
                logger.info(f"Found via email domain: {domain} → {parent}")
                return result
        
        # No match found
        result.parent_company = 'UNKNOWN'
        result.confidence = 0.0
        result.source = 'unknown'
        
        return result
    
    def extract_all_signals(self, text: str) -> Dict:
        """
        Extract all parent company signals for analysis.
        
        Returns all patterns found, even if primary match found.
        Useful for debugging and validation.
        """
        signals = {
            'ir_submitter': None,
            'ir_submitter_inr': None,
            'co_addresses': [],
            'email_domains': [],
            'final_parent': 'UNKNOWN',
            'final_confidence': 0.0,
            'final_source': 'unknown',
        }
        
        # IR Submitter
        ir_match = self.IR_SUBMITTER_PATTERN.search(text)
        if ir_match:
            signals['ir_submitter_inr'] = ir_match.group(1)
            raw = ir_match.group(2).strip()
            raw = re.sub(r'\s*\.\s*$', '', raw)
            signals['ir_submitter'] = raw
        
        # c/o Addresses (all of them)
        signals['co_addresses'] = [m.strip() for m in self.CO_ADDRESS_PATTERN.findall(text)]
        
        # Email domains (non-utility)
        emails = self.EMAIL_PATTERN.findall(text)
        signals['email_domains'] = [
            d.lower() for d in set(emails) 
            if d.lower() not in self.UTILITY_DOMAINS
        ]
        
        # Compute final
        result = self.extract(text)
        signals['final_parent'] = result.parent_company
        signals['final_confidence'] = result.confidence
        signals['final_source'] = result.source
        
        return signals


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_parent_company(text: str) -> Tuple[str, float, str]:
    """
    Quick extraction of parent company.
    
    Args:
        text: Full SGIA text
    
    Returns:
        Tuple of (parent_company, confidence, source)
    """
    extractor = ParentCompanyExtractor()
    result = extractor.extract(text)
    return result.parent_company, result.confidence, result.source


def batch_extract_parents(texts: List[str]) -> List[ParentExtractionResult]:
    """
    Extract parent companies from multiple documents.
    
    Args:
        texts: List of SGIA full texts
    
    Returns:
        List of ParentExtractionResult
    """
    extractor = ParentCompanyExtractor()
    return [extractor.extract(text) for text in texts]


# =============================================================================
# INTEGRATION WITH SMART EXTRACTOR
# =============================================================================

"""
Integration example for smart_extractor.py:

```python
from .parent_extractor import extract_parent_company

def extract(self, pdf_path: Path) -> ExtractionResult:
    # ... existing code ...
    
    # After extracting text sections
    full_text = "\\n".join(sections.values())
    
    # Extract parent using regex (fast, no LLM cost)
    parent, confidence, source = extract_parent_company(full_text)
    
    # Use regex result if confident, else fall back to LLM extraction
    if confidence >= 0.7:
        result.parent_company = parent
        result.parent_company_source = source
    else:
        # Use LLM extraction as fallback
        result.parent_company = extracted.get('parent_company_from_pdf', 'UNKNOWN')
        result.parent_company_source = 'llm'
    
    # ... rest of code ...
```
"""


# =============================================================================
# TEST CASES
# =============================================================================

if __name__ == '__main__':
    # Test with real SGIA text patterns
    test_cases = [
        # IR Submitter pattern
        """
        Transmission Service Provider shall interconnect Generator's Plant with Transmission 
        Service Provider's System consistent with the results of the Full Interconnection Study 
        that was prepared in response to generation interconnection request #24INR0485 to ERCOT 
        from Samsung C&T America, Inc.
        """,
        
        # c/o pattern
        """
        If to Interconnection Customer:
        Company Name: Rutile BESS, LLC
        c/o NextEra Energy Resources, LLC
        Attn: Manager, Interconnection
        """,
        
        # Email pattern
        """
        E-mail: kospo-yg@kospo.co.kr
        Phone: 713-261-5906
        """,
    ]
    
    extractor = ParentCompanyExtractor()
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        
        result = extractor.extract(text)
        print(f"Parent: {result.parent_company}")
        print(f"Confidence: {result.confidence}")
        print(f"Source: {result.source}")
        
        if result.ir_submitter_raw:
            print(f"IR Submitter: {result.ir_submitter_raw} (INR: {result.ir_submitter_inr})")
        if result.co_company_raw:
            print(f"c/o: {result.co_company_raw}")
        if result.email_domains:
            print(f"Email domains: {result.email_domains}")
