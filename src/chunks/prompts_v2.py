"""
Prompts V2 - Production-Grade Competitive Intelligence Extraction

Improvements over V1:
1. Few-shot examples from REAL SGIAs (not synthetic)
2. IR Submitter pattern for parent company (most reliable)
3. Chain-of-thought reasoning
4. Negative examples (what NOT to do)
5. Validation hints for common errors

Author: Santiago (UC3M Applied AI)
Date: December 2025
Version: 2.0
"""


# =============================================================================
# MAIN EXTRACTION PROMPT (PRODUCTION GRADE)
# =============================================================================

EXTRACTION_PROMPT_V2 = """You are an expert energy analyst extracting competitive intelligence from ERCOT Standard Generation Interconnection Agreements (SGIAs).

## YOUR TASK
Extract specific fields from this SGIA document. Focus on business-critical data that enables cost benchmarking.

## FIELD DEFINITIONS

### 1. SECURITY AMOUNTS (MOST IMPORTANT)
SGIAs require developers to post financial security in TWO phases:

- **security_design_usd**: Posted during engineering/study phase
  - Typical range: $500,000 - $5,000,000
  - Look for: "Design Phase", "Study Phase", "engineering security"

- **security_construction_usd**: Posted before construction begins
  - Typical range: $2,000,000 - $50,000,000
  - Look for: "Construction Phase", "construction security"

- **security_total_usd**: ONLY extract if document states an explicit total
  - If not stated, leave as 0 (we calculate in post-processing)

### 2. TRANSMISSION SERVICE PROVIDER (TSP)
The utility that owns the transmission infrastructure. Common TSPs:
- Electric Transmission Texas, LLC (ETT)
- Oncor Electric Delivery Company LLC
- CenterPoint Energy Houston Electric, LLC
- AEP Texas Inc.
- LCRA Transmission Services Corporation
- Sharyland Utilities, LP

### 3. PARENT COMPANY (from two sources)

**PRIMARY SOURCE - IR Submitter (in recitals, most reliable):**
Look for: "interconnection request #XXINRXXXX to ERCOT from [COMPANY NAME]"
This reveals the ACTUAL corporate sponsor, not the SPV.

**SECONDARY SOURCE - Notice Address (in Exhibit D/E):**
Look for: "c/o [COMPANY NAME]" in the Interconnection Customer's address

### 4. AMENDMENT STATUS
- is_amended: true if document title contains "Amended and Restated"
- amendment_type: "original", "first_amended", "second_amended", "assignment"

---

## FEW-SHOT EXAMPLES

### EXAMPLE 1: Standard Security Format
```
INPUT TEXT:
"The Interconnection Customer shall provide the following security totaling Eight Million
Dollars ($8,000,000), comprised of (i) Design Phase security in the amount of Two Million
Five Hundred Thousand Dollars ($2,500,000), and (ii) Construction Phase security in the
amount of Five Million Five Hundred Thousand Dollars ($5,500,000)."

OUTPUT:
{
  "security_design_usd": 2500000,
  "security_construction_usd": 5500000,
  "security_total_usd": 8000000,
  "extraction_notes": "Explicit total stated with both components"
}
```

### EXAMPLE 2: Table Format (no explicit total)
```
INPUT TEXT:
"EXHIBIT E - SECURITY ARRANGEMENT DETAILS
Phase               Amount
Design Phase        $1,500,000
Construction Phase  $4,200,000"

OUTPUT:
{
  "security_design_usd": 1500000,
  "security_construction_usd": 4200000,
  "security_total_usd": 0,
  "extraction_notes": "Table format, no explicit total stated"
}
```

### EXAMPLE 3: IR Submitter Pattern (Parent Company)
```
INPUT TEXT (from recitals):
"...Transmission Service Provider shall interconnect Generator's Plant with Transmission
Service Provider's System consistent with the results of the Full Interconnection Study
that was prepared in response to generation interconnection request #24INR0485 to ERCOT
from Samsung C&T America, Inc."

OUTPUT:
{
  "ir_submitter": "Samsung C&T America, Inc.",
  "parent_company_from_pdf": "SAMSUNG C&T"
}
```

### EXAMPLE 4: c/o Pattern (Secondary Parent Source)
```
INPUT TEXT (from Exhibit D):
"If to Interconnection Customer:
Company Name: Rutile BESS, LLC
c/o American Electric Power Service Corporation
Attn: Young Gyun Jeon"

OUTPUT:
{
  "parent_company_from_pdf": "",
  "extraction_notes": "c/o shows service company (AEP), not parent. Use IR submitter if available."
}
```

### EXAMPLE 5: TSP Identification
```
INPUT TEXT (from header):
"ERCOT STANDARD GENERATION INTERCONNECTION AGREEMENT
between
Electric Transmission Texas, LLC
And
Desert Sky Solar, LLC"

OUTPUT:
{
  "tsp_name": "Electric Transmission Texas, LLC",
  "tsp_normalized": "ETT"
}
```

---

## COMMON ERRORS TO AVOID

❌ DON'T extract amounts from "estimated costs" sections - those are TSP's construction costs, NOT security
❌ DON'T confuse "Letter of Credit amount" with total security - LC is just one form of security
❌ DON'T extract the TSP's internal contact as parent company
❌ DON'T leave security_total_usd as 0 if an explicit total IS stated
❌ DON'T extract security amounts from amendments if they're referencing the original agreement

---

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown, no explanation, no preamble):

{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0,
  "tsp_name": "",
  "tsp_normalized": "",
  "ir_submitter": "",
  "parent_company_from_pdf": "",
  "is_amended": false,
  "amendment_type": "",
  "commercial_operation_date": "",
  "extraction_confidence": 0.0,
  "extraction_notes": ""
}

## RULES
1. Convert word numbers: "Eight Million" = 8000000
2. Remove formatting: "$8,500,000" = 8500000
3. Extract BOTH security phases if they exist
4. Only fill security_total_usd if EXPLICITLY stated
5. Normalize TSP to standard abbreviation (ETT, ONCOR, CENTERPOINT, AEP, LCRA, etc.)
6. Set confidence 0.0-1.0 based on clarity of source text
7. Use extraction_notes to flag any ambiguity

---

## DOCUMENT TEXT:
"""


# =============================================================================
# RETRY PROMPT - SECURITY (with reasoning)
# =============================================================================

RETRY_SECURITY_PROMPT_V2 = """The security amounts were NOT found in the first extraction attempt.

This document MUST contain security information (it's legally required in SGIAs).

## STEP-BY-STEP SEARCH

Step 1: Search for "Exhibit E" or "EXHIBIT E" (Security Arrangement Details)
Step 2: Search for "Security" within 200 characters of dollar amounts
Step 3: Search for "Million Dollars" or "$X,XXX,XXX" patterns
Step 4: Search for table structures with "Amount" column

## PATTERNS TO MATCH

Pattern A: "totaling [NUMBER] [Million] Dollars ($X,XXX,XXX)"
Pattern B: "Design Phase... $X... Construction Phase... $Y"
Pattern C: Tables with "Phase" and "Amount" columns
Pattern D: "Letter of Credit" with amounts (this IS security)
Pattern E: "Interconnection Customer's Security" section

## TYPICAL RANGES (for validation)
- Design phase: $500,000 - $5,000,000
- Construction phase: $2,000,000 - $50,000,000
- Total: $2,500,000 - $55,000,000

If you find amounts OUTSIDE these ranges, double-check before extracting.

## OUTPUT

Return JSON with your reasoning:

{
  "security_design_usd": 0,
  "security_construction_usd": 0,
  "security_total_usd": 0,
  "found_in_section": "where you found it (e.g., 'Exhibit E', 'page 45')",
  "exact_quote": "the exact text containing the amounts (max 200 chars)",
  "confidence": 0.0
}

## DOCUMENT TEXT:
"""


# =============================================================================
# PARENT COMPANY EXTRACTION (Specialized)
# =============================================================================

PARENT_COMPANY_PROMPT = """Extract the REAL parent company (not the SPV) from this SGIA.

## PRIORITY ORDER

1. **IR SUBMITTER** (highest reliability - in recitals section)
   Pattern: "interconnection request #XXINRXXXX to ERCOT from [COMPANY]"
   This is the company that SUBMITTED the interconnection request to ERCOT.
   Example: "...request #24INR0485 to ERCOT from Samsung C&T America, Inc."
   → Parent: "SAMSUNG C&T"

2. **c/o ADDRESS** (secondary - in Exhibit D notices)
   Pattern: "c/o [COMPANY NAME]" in Interconnection Customer address
   Example: "c/o NextEra Energy Resources, LLC"
   → Parent: "NEXTERA"

3. **EMAIL DOMAIN** (tertiary - any @company.com emails)
   Pattern: Contact emails from non-utility domains
   Example: "kospo-yg@kospo.co.kr"
   → Hints at: "KOSPO" (Korean utility)

## IMPORTANT DISTINCTIONS

✅ IR Submitter = Corporate sponsor who filed with ERCOT (BEST source)
✅ c/o Company = Usually the parent's service company
❌ SPV Name = Just the project shell company (e.g., "Blue Sky Solar, LLC")
❌ TSP Contacts = Utility employees, not the developer

## OUTPUT

{
  "ir_submitter_raw": "",
  "ir_submitter_normalized": "",
  "co_address_company": "",
  "email_domains": [],
  "parent_company_final": "",
  "confidence": 0.0,
  "reasoning": ""
}

## DOCUMENT TEXT:
"""


# =============================================================================
# EQUIPMENT EXTRACTION (Solar/Wind/Battery)
# =============================================================================

EQUIPMENT_PROMPT_V2 = """Extract equipment specifications from this SGIA (Exhibit C section).

## SOLAR PROJECTS
- inverter_manufacturer: SUNGROW, SMA, GE, POWER ELECTRONICS, TMEIC, etc.
- inverter_model: Model number (e.g., "SG3125HV", "PVS980-58")
- inverter_quantity: Integer count
- inverter_capacity_kw: Individual unit rating if stated

## WIND PROJECTS
- turbine_manufacturer: Vestas, GE, Siemens Gamesa, Nordex, Goldwind
- turbine_model: Model number (e.g., "V150-4.2", "SG 5.0-145")
- turbine_quantity: Integer count
- turbine_capacity_mw: Individual unit rating if stated

## BATTERY PROJECTS
- inverter_manufacturer: Often the same companies as solar
- battery_manufacturer: CATL, BYD, Samsung SDI, LG Energy, Tesla
- storage_capacity_mwh: Total MWh if stated

## ALL PROJECTS
- poi_substation: Point of Interconnection substation name
- voltage_kv: Interconnection voltage (typically 138, 345)

## OUTPUT

{
  "project_type": "solar|wind|battery|hybrid",
  "inverter_manufacturer": "",
  "inverter_model": "",
  "inverter_quantity": 0,
  "turbine_manufacturer": "",
  "turbine_model": "",
  "turbine_quantity": 0,
  "battery_manufacturer": "",
  "storage_capacity_mwh": 0,
  "poi_substation": "",
  "voltage_kv": 0,
  "confidence": 0.0
}

## DOCUMENT TEXT:
"""


# =============================================================================
# VALIDATION PROMPT (Quality Check)
# =============================================================================

VALIDATION_PROMPT = """Review this extraction result for errors.

## EXTRACTED DATA
{extracted_json}

## VALIDATION CHECKS

1. **Security Reasonableness**
   - Design phase should be $500K - $5M
   - Construction phase should be $2M - $50M
   - Total should be sum of parts (or close)
   - security_per_kw should be $20-100/kW for most projects

2. **TSP Consistency**
   - TSP should match the utility territory where county is located
   - ETT = much of Texas, ONCOR = Dallas area, CENTERPOINT = Houston

3. **Date Logic**
   - COD should be in the future or recent past
   - COD should be after IA Signed date

4. **Parent Company**
   - Should NOT be an SPV name (e.g., "Blue Sky Solar, LLC")
   - Should be a known developer or "UNKNOWN"

## OUTPUT

{
  "is_valid": true,
  "issues": [],
  "suggested_corrections": {},
  "confidence_adjustment": 0.0
}
"""


# =============================================================================
# PROMPT SELECTOR (based on document characteristics)
# =============================================================================

def get_optimal_prompt(
    text_quality: float,
    is_scanned: bool,
    has_tables: bool,
    document_length: int,
) -> str:
    """
    Select the optimal prompt based on document characteristics.

    Args:
        text_quality: 0-1 score of OCR/text quality
        is_scanned: True if document appears to be scanned
        has_tables: True if document contains table structures
        document_length: Character count

    Returns:
        Appropriate prompt string
    """
    # For scanned/low-quality docs, use simpler prompt with more examples
    if is_scanned or text_quality < 0.5:
        return EXTRACTION_PROMPT_V2  # Vision-friendly

    # For high-quality text docs, use full prompt
    return EXTRACTION_PROMPT_V2


# =============================================================================
# PROMPT TEMPLATES FOR SPECIFIC SECTIONS
# =============================================================================

SECTION_PROMPTS = {
    'exhibit_e': """Focus on EXHIBIT E - SECURITY ARRANGEMENT DETAILS.
Extract:
- Design Phase security amount
- Construction Phase security amount
- Total security (only if explicit)
- Letter of Credit details if present

{base_prompt}
""",

    'recitals': """Focus on the RECITALS section (first 2-3 pages after cover letter).
Extract:
- IR Submitter (from "interconnection request #XXINRXXXX to ERCOT from [COMPANY]")
- TSP name (from "between [TSP] and [Generator]")
- Amendment status

{base_prompt}
""",

    'exhibit_c': """Focus on EXHIBIT C - INTERCONNECTION DETAILS.
Extract:
- Equipment specifications (inverters/turbines)
- POI Substation name
- Voltage level
- Generating unit details

{base_prompt}
""",
}
