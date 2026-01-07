#!/usr/bin/env python3
"""
Post-Processing Enrichment Pipeline
====================================

This script enriches raw extraction results with additional intelligence:

1. MERGE STRATEGY
   - Combines V5.2 (text-based) and V5.3 (vision-based) extraction results
   - Takes best security value from either run per document
   - Rationale: Vision extraction cost $40 more but achieved WORSE results (77 vs 79),
     however some individual documents extracted better with vision

2. EMAIL DOMAIN → PARENT COMPANY MAPPING
   - SGIAs contain contact emails in cover letters
   - Email domains reveal true parent companies behind SPV legal entities
   - Example: aoprenewables.com → Alpha Omega Power (not "GAMAY ENERGY STORAGE LLC")
   - Novel technique: No public database captures this parent-subsidiary relationship

3. ZONE/FUEL MEDIAN ESTIMATION
   - For 53 documents missing security amounts, estimate using corpus statistics
   - Calculate median $/kW by (zone, fuel_type) from documents with actual values
   - Apply: estimated_security = median_per_kw * capacity_mw * 1000
   - Enables 91% queryable coverage vs 60% with actual-only

4. SPV NAME PATTERN MATCHING
   - Some SPV names contain developer identifiers
   - Example: "RWE Renewables Development, LLC" → parent = RWE
   - Cross-reference with GIS Interconnecting Entity field

Results:
- Security coverage: 60% actual → 91% queryable (actual + estimated)
- Parent company: 22% → 47% real parent identification
- 47 unique developers identified across 133 documents

Author: Santiago (UC3M Applied AI Master's)
Date: December 2025
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# EMAIL DOMAIN → PARENT COMPANY MAPPING
# =============================================================================
# Discovered by analyzing email domains extracted from SGIA cover letters
# These reveal the TRUE parent companies behind SPV legal entities

EMAIL_TO_PARENT = {
    # Major Developers
    'aoprenewables.com': 'ALPHA OMEGA POWER (AOP)',
    'sunchasepower.com': 'SUNCHASE POWER',
    'rwe.com': 'RWE',
    'totalenergies.com': 'TOTALENERGIES',
    'acciona.com': 'ACCIONA',
    'ferrovial.com': 'FERROVIAL',
    'nexusrenewables.ca': 'NEXUS RENEWABLES',
    'x-elio.com': 'X-ELIO',
    'tesla.com': 'TESLA',
    'nrg.com': 'NRG ENERGY',

    # Infrastructure/Utilities
    'centerpointenergy.com': 'CENTERPOINT ENERGY',
    'cpsenergy.com': 'CPS ENERGY',

    # Smaller Developers
    'adapturerenewables.com': 'ADAPTURE RENEWABLES',
    'ocienergy.com': 'OCI ENERGY',
    'rocklandcapital.com': 'ROCKLAND CAPITAL',
    'navitaspower.net': 'NAVITAS ENERGY',
    'hfresco.com': 'H. FRESCO ENERGY',
    'brprenewables.com': 'BRP RENEWABLES',
    'granitesp.com': 'GRANITE SOURCE POWER',
    'hecateenergy.com': 'HECATE ENERGY',
    'greenbeltrenewables.com': 'GREENBELT RENEWABLES',
    'eolianenergy.com': 'EOLIAN ENERGY',
    'diodeventures.com': 'DIODE VENTURES',
    'naturgy.com': 'NATURGY',
    'nightpeak.energy': 'NIGHTPEAK ENERGY',
    'solarproponent.com': 'SOLAR PROPONENT',
    'peregrineenergysolutions.com': 'PEREGRINE ENERGY',
}

# SPV name patterns that reveal parent companies
SPV_PATTERNS = {
    'rwe': 'RWE',
    'nextera': 'NEXTERA ENERGY',
    'terra-gen': 'TERRA-GEN',
    'intersect': 'INTERSECT POWER',
    'swift current': 'SWIFT CURRENT ENERGY',
    'apex': 'APEX CLEAN ENERGY',
    'invenergy': 'INVENERGY',
    'edf': 'EDF RENEWABLES',
    'engie': 'ENGIE',
    'longroad': 'LONGROAD ENERGY',
}


def merge_extraction_results(v52_path: Path, v53_path: Path) -> List[Dict]:
    """
    Merge V5.2 and V5.3 extraction results, taking best security value.

    Strategy:
    - Use V5.2 as base (better overall: 79/133 vs 77/133)
    - For each document, if V5.3 has security and V5.2 doesn't, use V5.3 value
    """
    with open(v52_path) as f:
        v52 = {r['filename']: r for r in json.load(f)}

    with open(v53_path) as f:
        v53 = {r['filename']: r for r in json.load(f)}

    merged = []
    upgrades = 0

    for filename, record in v52.items():
        # Check if V5.3 has security and V5.2 doesn't
        if not record.get('security_total_usd') and filename in v53:
            v53_security = v53[filename].get('security_total_usd')
            if v53_security:
                record['security_total_usd'] = v53_security
                record['security_per_kw'] = v53[filename].get('security_per_kw')
                record['merge_source'] = 'v53'
                upgrades += 1

        merged.append(record)

    print(f"Merged results: {upgrades} documents upgraded from V5.3")
    return merged


def enrich_parent_from_email(records: List[Dict]) -> int:
    """
    Identify parent companies from email domains in extraction results.

    Returns number of parent companies identified.
    """
    identified = 0

    for record in records:
        parent = record.get('parent_company', '')

        # Skip if already has real parent (not LLC/SPV)
        if parent and 'LLC' not in parent.upper() and parent != 'UNKNOWN':
            continue

        # Check email domains
        domains = record.get('email_domains_found', []) or []
        for domain in domains:
            clean_domain = domain.replace('@', '').lower()
            if clean_domain in EMAIL_TO_PARENT:
                record['parent_company'] = EMAIL_TO_PARENT[clean_domain]
                record['parent_evidence'] = f'email_domain:{clean_domain}'
                identified += 1
                break

    print(f"Parent companies identified from email: {identified}")
    return identified


def enrich_parent_from_spv_pattern(records: List[Dict]) -> int:
    """
    Identify parent companies from SPV name patterns.
    """
    identified = 0

    for record in records:
        parent = record.get('parent_company', '')

        if parent and 'LLC' not in parent.upper() and parent != 'UNKNOWN':
            continue

        spv = (record.get('developer_spv', '') or '').lower()
        ir = (record.get('ir_submitter', '') or '').lower()

        for pattern, parent_name in SPV_PATTERNS.items():
            if pattern in spv or pattern in ir:
                record['parent_company'] = parent_name
                record['parent_evidence'] = f'spv_pattern:{pattern}'
                identified += 1
                break

    print(f"Parent companies identified from SPV patterns: {identified}")
    return identified


def estimate_missing_security(records: List[Dict]) -> int:
    """
    Estimate security amounts for documents without actual values.

    Uses median $/kW by (zone, fuel_type) from documents with actual values.
    Only estimates when we have >= 3 samples for statistical validity.
    """
    # Calculate medians from actual values
    zone_fuel_values = defaultdict(list)

    for r in records:
        zone = r.get('zone', '')
        fuel = r.get('fuel_type', '')
        spk = r.get('security_per_kw', 0) or 0

        # Filter outliers (typical range is $10-150/kW)
        if 10 < spk < 150:
            zone_fuel_values[(zone, fuel)].append(spk)

    # Calculate medians where we have enough samples
    medians = {}
    for key, values in zone_fuel_values.items():
        if len(values) >= 3:
            medians[key] = statistics.median(values)

    print(f"Zone/Fuel medians calculated: {len(medians)} combinations")
    for (zone, fuel), median in sorted(medians.items()):
        print(f"  {zone}/{fuel}: ${median:.2f}/kW")

    # Apply estimates
    estimated = 0
    for r in records:
        if r.get('security_total_usd'):
            continue  # Has actual value

        zone = r.get('zone', '')
        fuel = r.get('fuel_type', '')
        capacity = r.get('capacity_mw', 0) or 0

        key = (zone, fuel)
        if key in medians and capacity > 0:
            estimated_usd = medians[key] * capacity * 1000
            r['security_estimated_usd'] = round(estimated_usd, 2)
            r['security_estimated_per_kw'] = round(medians[key], 2)
            r['security_estimate_source'] = f'Zone/Fuel median: {zone}/{fuel}'
            estimated += 1

    print(f"Security estimated for {estimated} documents")
    return estimated


def run_enrichment(
    v52_path: Path,
    v53_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> List[Dict]:
    """
    Run full enrichment pipeline.
    """
    print("=" * 70)
    print("POST-PROCESSING ENRICHMENT PIPELINE")
    print("=" * 70)

    # Step 1: Load or merge
    if v53_path and v53_path.exists():
        print("\n1. Merging V5.2 + V5.3 results...")
        records = merge_extraction_results(v52_path, v53_path)
    else:
        print("\n1. Loading V5.2 results...")
        with open(v52_path) as f:
            records = json.load(f)

    # Step 2: Email domain enrichment
    print("\n2. Enriching parent companies from email domains...")
    enrich_parent_from_email(records)

    # Step 3: SPV pattern enrichment
    print("\n3. Enriching parent companies from SPV patterns...")
    enrich_parent_from_spv_pattern(records)

    # Step 4: Security estimation
    print("\n4. Estimating missing security amounts...")
    estimate_missing_security(records)

    # Summary
    n = len(records)
    actual = sum(1 for r in records if r.get('security_total_usd'))
    estimated = sum(1 for r in records if r.get('security_estimated_usd'))
    real_parents = sum(1 for r in records
                      if r.get('parent_company')
                      and 'LLC' not in r.get('parent_company', '').upper()
                      and r.get('parent_company') != 'UNKNOWN')

    print("\n" + "=" * 70)
    print("ENRICHMENT SUMMARY")
    print("=" * 70)
    print(f"Total documents:              {n}")
    print(f"Security (actual):            {actual}/{n} ({100*actual/n:.1f}%)")
    print(f"Security (estimated):         {estimated}/{n}")
    print(f"Security (queryable):         {actual + estimated}/{n} ({100*(actual+estimated)/n:.1f}%)")
    print(f"Real parent companies:        {real_parents}/{n} ({100*real_parents/n:.1f}%)")

    # Save
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return records


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Enrich extraction results')
    parser.add_argument('--v52', type=Path, required=True, help='V5.2 results JSON')
    parser.add_argument('--v53', type=Path, help='V5.3 results JSON (optional)')
    parser.add_argument('--output', type=Path, help='Output path')

    args = parser.parse_args()

    run_enrichment(args.v52, args.v53, args.output)
