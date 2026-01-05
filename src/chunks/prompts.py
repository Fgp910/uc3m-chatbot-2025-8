"""
Prompts - Competitive Intelligence Extraction (CEO-GRADE)

Focus on high-value fields that can't be obtained from GIS metadata:
1. Security amounts (design + construction phases separately)
2. TSP name (for cost-by-utility analysis)
3. TRUE Parent company (the hidden intel - NOT the SPV shell)

Uses split security strategy to prevent under-reporting costs.
Uses IR submitter + email domain patterns to unmask hidden ownership.

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""


# =============================================================================
# MAIN EXTRACTION PROMPT - CEO GRADE
# =============================================================================

SECURITY_EXTRACTION_PROMPT = """You are extracting competitive intelligence from an ERCOT SGIA document.

## PRIORITY 1: SECURITY AMOUNTS (Exhibit D or E)

SGIAs often list TWO separate security amounts. Extract BOTH:

1. **security_design_usd**: Design/Study/Procurement phase security
   - Typically $1-5 Million
   - Look for: "Design Phase", "Study Phase", "pre-construction"
   
2. **security_construction_usd**: Construction phase security  
   - Typically $5-50 Million
   - Look for: "Construction Phase", "Construction Security"

3. **security_total_usd**: ONLY if an explicit total is stated
   - Look for: "totaling X Million Dollars ($Y,000,000)"
   - If not explicitly stated, leave as 0 (we calculate in Python)

## PRIORITY 2: TRUE PARENT COMPANY (THE HIDDEN INTEL)

The "Interconnection Customer" (Generator) is usually a SHELL COMPANY like "Rutile BESS, LLC".
Your job is to find the REAL parent/sponsor. Look in these places:

**A) RECITALS (First 2 pages):**
   - Pattern: "interconnection request #XXINRXXXX to ERCOT from [REAL COMPANY]"
   - Example: "request from Samsung C&T America, Inc." → parent = "SAMSUNG"

**B) EXHIBIT E - NOTICES (Contact Section):**
   - Email domains reveal the parent: "@nexteraenergy.com" → "NEXTERA"
   - Email: "@kospo.co.kr" → "KOSPO" (Korea South-East Power)
   - Email: "@rwe.com" → "RWE"
   - "c/o [COMPANY NAME]" references: "c/o NextEra Energy Resources" → "NEXTERA"

**C) ADDRESS CLUES:**
   - "700 Universe Blvd, Juno Beach, FL" = NextEra headquarters
   - Korean contact names + .co.kr emails = Korean consortium

## PRIORITY 3: TSP NAME

- Full legal name of the Transmission Service Provider
- Examples: "Electric Transmission Texas, LLC", "Oncor Electric Delivery", "LCRA TSC"

## PRIORITY 4: AMENDMENT STATUS

- Is this "Amended and Restated"?

Return ONLY valid JSON (no markdown):
{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0,
  "tsp_name": "",
  "generator_spv": "",
  "ir_submitter": "",
  "parent_company_inferred": "",
  "parent_evidence": "",
  "email_domains_found": [],
  "is_amended": false,
  "extraction_confidence": 0.0
}

FIELD EXPLANATIONS:
- generator_spv: The shell LLC name (e.g., "Rutile BESS, LLC")
- ir_submitter: Company that submitted the IR (from recitals) - often the REAL parent
- parent_company_inferred: Your best guess at the true parent (e.g., "SAMSUNG", "NEXTERA")
- parent_evidence: Quote the text that revealed the parent
- email_domains_found: List of non-TSP email domains (e.g., ["kospo.co.kr", "samsung.com"])

DOCUMENT TEXT:
"""


# =============================================================================
# COMBINED EXTRACTION (Single-call for efficiency)
# =============================================================================

COMBINED_EXTRACTION_PROMPT = """You are extracting competitive intelligence from an ERCOT SGIA document.

Focus on these high-value fields (ignore standard legal boilerplate in Articles 1-10):

## PRIORITY 1 - Security (MUST extract both components):
- security_design_usd: Design/study phase (typically $1-5M)
- security_construction_usd: Construction phase (typically $5-50M)
- security_total_usd: ONLY if explicitly stated (otherwise 0)

## PRIORITY 2 - TRUE PARENT COMPANY (THE HIDDEN INTEL):
The "Generator" is usually a shell LLC. Find the REAL owner by looking for:
- RECITALS: "interconnection request from [REAL COMPANY]..."
- EXHIBIT E emails: @nexteraenergy.com, @rwe.com, @kospo.co.kr reveal parent
- "c/o [PARENT COMPANY]" in addresses

## PRIORITY 3 - TSP & Status:
- tsp_name: Transmission Service Provider (utility)
- is_amended: true if "Amended and Restated"

## PRIORITY 4 - Timeline (from Exhibit B):
- commercial_operation_date: COD in YYYY-MM-DD format

Return ONLY valid JSON:
{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0,
  "tsp_name": "",
  "generator_spv": "",
  "ir_submitter": "",
  "parent_company_inferred": "",
  "parent_evidence": "",
  "email_domains_found": [],
  "is_amended": false,
  "amendment_type": "",
  "commercial_operation_date": "",
  "extraction_confidence": 0.0
}

RULES:
1. Convert "Eight Million Dollars" = 8000000
2. Extract BOTH security_design_usd AND security_construction_usd
3. ir_submitter is from recitals: "request from [COMPANY]"
4. parent_company_inferred = your best guess (NEXTERA, RWE, SAMSUNG, etc.)
5. Dates in YYYY-MM-DD format

DOCUMENT TEXT:
"""


# =============================================================================
# VISION PROMPT (for scanned PDFs)
# =============================================================================

VISION_EXTRACTION_PROMPT = """Analyze these SGIA document pages (may be scanned images).

Extract these critical fields:

1. SECURITY AMOUNTS (look for dollar amounts):
   - Design phase security (smaller, $1-5M)
   - Construction phase security (larger, $5-50M)
   - Look in tables for "Million Dollars"

2. TRUE PARENT COMPANY:
   - Look in first 2 pages for: "interconnection request from [COMPANY]"
   - Look in contact section for email domains (reveals parent)
   - "@nexteraenergy.com" = NextEra, "@rwe.com" = RWE, etc.

3. TSP NAME (in header or cover letter)

4. Check if "Amended and Restated"

Return ONLY valid JSON:
{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0,
  "tsp_name": "",
  "ir_submitter": "",
  "parent_company_inferred": "",
  "email_domains_found": [],
  "is_amended": false,
  "extraction_confidence": 0.0
}

Convert all dollar amounts to integers.
"Eight Million" = 8000000
"""


# =============================================================================
# RETRY PROMPTS (for missing critical fields)
# =============================================================================

RETRY_SECURITY_PROMPT = """The security amounts were NOT found in the first extraction.

Search this text VERY CAREFULLY for dollar amounts:

PATTERNS TO FIND:
1. "totaling [Word] Million Dollars ($X,XXX,XXX)"
2. "Design Phase... Amount of $X"
3. "Construction Phase... Amount of $Y"
4. Tables with "Security" and dollar amounts
5. "Letter of Credit" amounts

Typical ranges:
- Design: $1M - $5M
- Construction: $5M - $50M

Return ONLY valid JSON:
{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0,
  "found_text": "quote the exact text"
}

DOCUMENT TEXT:
"""


RETRY_TSP_PROMPT = """The TSP name was NOT found. Search for the Transmission Service Provider:

Common TSPs:
- Electric Transmission Texas, LLC (ETT)
- Oncor Electric Delivery Company LLC
- CenterPoint Energy Houston Electric, LLC
- LCRA Transmission Services Corporation
- AEP Texas Inc.
- Sharyland Utilities, LP

Look in header or "between [TSP] and [Generator]"

Return ONLY valid JSON:
{
  "tsp_name": "",
  "found_in": "where you found it"
}

DOCUMENT TEXT:
"""


RETRY_PARENT_PROMPT = """The parent company was NOT found. Look carefully for the TRUE OWNER.

SEARCH FOR:
1. In RECITALS: "interconnection request from [COMPANY]" or "IR submitted by [COMPANY]"
2. In NOTICES: Email domains (not the TSP's domain)
   - @nexteraenergy.com = NextEra
   - @rwe.com = RWE  
   - @invenergy.com = Invenergy
   - @edpr.com = EDP
   - @samsung.com = Samsung
   - Korean domains (.co.kr) = Korean consortium
3. "c/o [COMPANY]" in addresses

Return ONLY valid JSON:
{
  "ir_submitter": "",
  "parent_company_inferred": "",
  "email_domains_found": [],
  "evidence": "quote the text"
}

DOCUMENT TEXT:
"""
