"""
Validation Module - Production-Grade Data Quality

Features:
1. Range validation for security amounts
2. Anomaly detection with z-scores
3. Cross-field consistency checks
4. TSP-County territory validation
5. Data quality scoring

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from statistics import mean, stdev

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RULES
# =============================================================================

VALIDATION_RULES = {
    # Security amount ranges (USD)
    'security_design_min': 100_000,       # $100K minimum
    'security_design_max': 10_000_000,    # $10M maximum
    'security_design_typical_min': 500_000,
    'security_design_typical_max': 5_000_000,
    
    'security_construction_min': 500_000,   # $500K minimum
    'security_construction_max': 100_000_000,  # $100M maximum
    'security_construction_typical_min': 2_000_000,
    'security_construction_typical_max': 50_000_000,
    
    'security_total_min': 500_000,
    'security_total_max': 150_000_000,
    
    # Security per kW ranges ($/kW)
    'security_per_kw_min': 5,     # $5/kW minimum
    'security_per_kw_max': 200,   # $200/kW maximum
    'security_per_kw_typical_min': 20,
    'security_per_kw_typical_max': 80,
    
    # Confidence thresholds
    'low_confidence_threshold': 0.5,
    'high_confidence_threshold': 0.8,
    
    # Anomaly detection
    'zscore_threshold': 2.5,  # Flag if more than 2.5 std devs from mean
}


# TSP-Territory mapping for validation
TSP_TERRITORIES = {
    'ETT': {
        'primary_zones': ['WEST', 'SOUTH', 'COAST', 'NORTH'],
        'counties': ['PECOS', 'REEVES', 'WARD', 'ECTOR', 'MIDLAND', 'HOWARD', 
                    'MARTIN', 'ANDREWS', 'CRANE', 'UPTON', 'WEBB', 'DIMMIT',
                    'MAVERICK', 'KINNEY', 'VAL VERDE', 'STARR', 'HIDALGO'],
    },
    'ONCOR': {
        'primary_zones': ['NORTH', 'CENTRAL'],
        'counties': ['DALLAS', 'TARRANT', 'COLLIN', 'DENTON', 'ELLIS', 'JOHNSON',
                    'KAUFMAN', 'ROCKWALL', 'HUNT', 'NAVARRO', 'HENDERSON', 'HILL'],
    },
    'CENTERPOINT': {
        'primary_zones': ['COAST'],
        'counties': ['HARRIS', 'FORT BEND', 'BRAZORIA', 'GALVESTON', 'MONTGOMERY',
                    'WALLER', 'LIBERTY', 'CHAMBERS'],
    },
    'AEP': {
        'primary_zones': ['WEST', 'SOUTH', 'CENTRAL'],
        'counties': ['TOM GREEN', 'RUNNELS', 'COLEMAN', 'BROWN', 'MCCULLOCH',
                    'CONCHO', 'MENARD', 'SCHLEICHER', 'IRION', 'SUTTON'],
    },
    'LCRA': {
        'primary_zones': ['CENTRAL', 'SOUTH'],
        'counties': ['TRAVIS', 'WILLIAMSON', 'HAYS', 'BASTROP', 'CALDWELL',
                    'FAYETTE', 'COLORADO', 'BLANCO', 'BURNET', 'LLANO'],
    },
}


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating an extraction."""
    
    is_valid: bool = True
    quality_score: float = 1.0  # 0-1 score
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    suggested_corrections: Dict = field(default_factory=dict)
    
    def add_issue(self, issue: str, severity: str = 'error'):
        """Add a validation issue."""
        if severity == 'error':
            self.issues.append(issue)
            self.is_valid = False
            self.quality_score -= 0.2
        elif severity == 'warning':
            self.warnings.append(issue)
            self.quality_score -= 0.1
        elif severity == 'anomaly':
            self.anomalies.append(issue)
            self.quality_score -= 0.05
        
        self.quality_score = max(0, self.quality_score)
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'quality_score': round(self.quality_score, 2),
            'issues': self.issues,
            'warnings': self.warnings,
            'anomalies': self.anomalies,
            'suggested_corrections': self.suggested_corrections,
        }


# =============================================================================
# VALIDATOR CLASS
# =============================================================================

class ExtractionValidator:
    """
    Validate extracted data for quality and consistency.
    
    Features:
    - Range validation
    - Cross-field consistency
    - TSP-territory matching
    - Anomaly detection (requires batch data)
    """
    
    def __init__(self):
        self.batch_stats = {}  # Populated after processing batch
    
    def validate_single(self, extraction: Dict) -> ValidationResult:
        """
        Validate a single extraction result.
        
        Args:
            extraction: Dict from ExtractionResult.to_dict()
        
        Returns:
            ValidationResult with issues and quality score
        """
        result = ValidationResult()
        
        # 1. Security amount validation
        self._validate_security(extraction, result)
        
        # 2. TSP validation
        self._validate_tsp(extraction, result)
        
        # 3. Cross-field consistency
        self._validate_consistency(extraction, result)
        
        # 4. Completeness check
        self._validate_completeness(extraction, result)
        
        return result
    
    def _validate_security(self, extraction: Dict, result: ValidationResult):
        """Validate security amounts are within reasonable ranges."""
        
        design = extraction.get('security_design_usd', 0) or 0
        construction = extraction.get('security_construction_usd', 0) or 0
        total = extraction.get('security_total_usd', 0) or 0
        per_kw = extraction.get('security_per_kw', 0) or 0
        capacity = extraction.get('capacity_mw', 0) or 0
        
        # Check design phase
        if design > 0:
            if design < VALIDATION_RULES['security_design_min']:
                result.add_issue(
                    f"Design security ${design:,.0f} below minimum ${VALIDATION_RULES['security_design_min']:,}",
                    'warning'
                )
            if design > VALIDATION_RULES['security_design_max']:
                result.add_issue(
                    f"Design security ${design:,.0f} above maximum ${VALIDATION_RULES['security_design_max']:,}",
                    'error'
                )
        
        # Check construction phase
        if construction > 0:
            if construction < VALIDATION_RULES['security_construction_min']:
                result.add_issue(
                    f"Construction security ${construction:,.0f} below minimum",
                    'warning'
                )
            if construction > VALIDATION_RULES['security_construction_max']:
                result.add_issue(
                    f"Construction security ${construction:,.0f} above maximum",
                    'error'
                )
        
        # Check total
        if total > 0:
            # Verify total = design + construction (with tolerance)
            expected_total = design + construction
            if expected_total > 0:
                diff_pct = abs(total - expected_total) / expected_total
                if diff_pct > 0.1:  # 10% tolerance
                    result.add_issue(
                        f"Total ${total:,.0f} doesn't match D+C ${expected_total:,.0f} ({diff_pct*100:.0f}% diff)",
                        'warning'
                    )
        
        # Check per kW
        if per_kw > 0:
            if per_kw < VALIDATION_RULES['security_per_kw_min']:
                result.add_issue(
                    f"Security ${per_kw:.2f}/kW unusually low (min ${VALIDATION_RULES['security_per_kw_min']}/kW)",
                    'warning'
                )
            if per_kw > VALIDATION_RULES['security_per_kw_max']:
                result.add_issue(
                    f"Security ${per_kw:.2f}/kW unusually high (max ${VALIDATION_RULES['security_per_kw_max']}/kW)",
                    'warning'
                )
        
        # Check if security is missing for large project
        if total == 0 and capacity >= 100:
            result.add_issue(
                f"No security extracted for {capacity:.0f} MW project",
                'error'
            )
    
    def _validate_tsp(self, extraction: Dict, result: ValidationResult):
        """Validate TSP matches expected territory."""
        
        tsp = extraction.get('tsp_normalized', '') or extraction.get('tsp_name', '')
        county = extraction.get('county', '').upper()
        zone = extraction.get('zone', '')
        
        if not tsp:
            result.add_issue("TSP not extracted", 'warning')
            return
        
        # Check if TSP is known
        tsp_upper = tsp.upper()
        if tsp_upper not in TSP_TERRITORIES:
            # Unknown TSP - just note it
            result.add_issue(f"Unknown TSP: {tsp}", 'warning')
            return
        
        # Check if county is in TSP's territory
        territory = TSP_TERRITORIES[tsp_upper]
        if county and county not in territory.get('counties', []):
            # Not necessarily wrong, just flag for review
            result.add_issue(
                f"County {county} not typical for TSP {tsp}",
                'anomaly'
            )
        
        # Check if zone matches
        if zone and zone not in territory.get('primary_zones', []):
            result.add_issue(
                f"Zone {zone} not typical for TSP {tsp}",
                'anomaly'
            )
    
    def _validate_consistency(self, extraction: Dict, result: ValidationResult):
        """Validate cross-field consistency."""
        
        # Check parent company vs SPV
        parent = extraction.get('parent_company', '')
        spv = extraction.get('developer_spv', '')
        
        if parent and parent != 'UNKNOWN':
            # Parent should NOT look like an SPV (no "LLC" at end, etc.)
            if 'LLC' in parent or 'LP' in parent or 'Inc' in parent:
                # Might be wrong - should be normalized (NEXTERA, not "NextEra Energy, LLC")
                result.add_issue(
                    f"Parent company '{parent}' looks like SPV name",
                    'warning'
                )
        
        # Check fuel type vs equipment
        fuel = extraction.get('fuel_type', '')
        inverter = extraction.get('inverter_manufacturer', '')
        turbine = extraction.get('turbine_manufacturer', '')
        
        if fuel == 'SOL' and turbine and not inverter:
            result.add_issue(
                "Solar project has turbine but no inverter",
                'warning'
            )
        if fuel == 'WIN' and inverter and not turbine:
            result.add_issue(
                "Wind project has inverter but no turbine",
                'warning'
            )
        
        # Check amendment status
        is_amended = extraction.get('is_amended', False)
        amendment_type = extraction.get('amendment_type', '')
        
        if is_amended and not amendment_type:
            result.add_issue(
                "Marked as amended but no amendment type specified",
                'warning'
            )
    
    def _validate_completeness(self, extraction: Dict, result: ValidationResult):
        """Check that critical fields are present."""
        
        critical_fields = [
            ('security_total_usd', 'Security amount'),
            ('tsp_name', 'TSP name'),
        ]
        
        important_fields = [
            ('parent_company', 'Parent company'),
            # COD removed - already in Gold Master CSV from GIS data
        ]
        
        for field_name, display_name in critical_fields:
            value = extraction.get(field_name)
            if not value or value == 0 or value == 'UNKNOWN':
                result.add_issue(f"Missing critical field: {display_name}", 'error')
        
        for field_name, display_name in important_fields:
            value = extraction.get(field_name)
            if not value or value == 0 or value == 'UNKNOWN':
                result.add_issue(f"Missing important field: {display_name}", 'warning')
    
    def compute_batch_stats(self, extractions: List[Dict]):
        """
        Compute statistics from a batch for anomaly detection.
        
        Call this after processing all documents.
        """
        # Security per kW by zone
        zone_security = {}
        for e in extractions:
            zone = e.get('zone', 'OTHER')
            spk = e.get('security_per_kw', 0)
            if spk > 0:
                if zone not in zone_security:
                    zone_security[zone] = []
                zone_security[zone].append(spk)
        
        self.batch_stats['zone_security'] = {}
        for zone, values in zone_security.items():
            if len(values) >= 3:
                self.batch_stats['zone_security'][zone] = {
                    'mean': mean(values),
                    'stdev': stdev(values) if len(values) > 1 else 0,
                    'count': len(values),
                }
        
        # Overall security per kW
        all_spk = [e.get('security_per_kw', 0) for e in extractions if e.get('security_per_kw', 0) > 0]
        if len(all_spk) >= 3:
            self.batch_stats['overall_security_per_kw'] = {
                'mean': mean(all_spk),
                'stdev': stdev(all_spk) if len(all_spk) > 1 else 0,
                'count': len(all_spk),
            }
        
        logger.info(f"Computed batch stats from {len(extractions)} extractions")
    
    def detect_anomalies(self, extraction: Dict) -> List[str]:
        """
        Detect anomalies using batch statistics.
        
        Call compute_batch_stats() first!
        """
        anomalies = []
        
        if 'overall_security_per_kw' not in self.batch_stats:
            return anomalies
        
        stats = self.batch_stats['overall_security_per_kw']
        spk = extraction.get('security_per_kw', 0)
        
        if spk > 0 and stats['stdev'] > 0:
            zscore = (spk - stats['mean']) / stats['stdev']
            
            if abs(zscore) > VALIDATION_RULES['zscore_threshold']:
                direction = "high" if zscore > 0 else "low"
                anomalies.append(
                    f"Security ${spk:.2f}/kW is {abs(zscore):.1f} std devs {direction} "
                    f"(mean=${stats['mean']:.2f}, stdev=${stats['stdev']:.2f})"
                )
        
        # Zone-specific anomaly
        zone = extraction.get('zone', '')
        if zone in self.batch_stats.get('zone_security', {}):
            zone_stats = self.batch_stats['zone_security'][zone]
            if spk > 0 and zone_stats['stdev'] > 0:
                zscore = (spk - zone_stats['mean']) / zone_stats['stdev']
                
                if abs(zscore) > VALIDATION_RULES['zscore_threshold']:
                    direction = "high" if zscore > 0 else "low"
                    anomalies.append(
                        f"For {zone} zone: ${spk:.2f}/kW is {abs(zscore):.1f} std devs {direction}"
                    )
        
        return anomalies
    
    def get_quality_report(self, validations: List[ValidationResult]) -> Dict:
        """Generate overall quality report from batch validations."""
        
        n = len(validations)
        if n == 0:
            return {'error': 'No validations provided'}
        
        valid_count = sum(1 for v in validations if v.is_valid)
        avg_score = mean(v.quality_score for v in validations)
        
        # Count issue types
        all_issues = []
        all_warnings = []
        all_anomalies = []
        for v in validations:
            all_issues.extend(v.issues)
            all_warnings.extend(v.warnings)
            all_anomalies.extend(v.anomalies)
        
        return {
            'total_extractions': n,
            'valid_extractions': valid_count,
            'valid_percentage': round(100 * valid_count / n, 1),
            'average_quality_score': round(avg_score, 2),
            'total_issues': len(all_issues),
            'total_warnings': len(all_warnings),
            'total_anomalies': len(all_anomalies),
            'grade': self._compute_grade(valid_count / n, avg_score),
        }
    
    def _compute_grade(self, valid_pct: float, avg_score: float) -> str:
        """Compute letter grade for extraction quality."""
        composite = (valid_pct + avg_score) / 2
        
        if composite >= 0.95:
            return 'A+'
        elif composite >= 0.90:
            return 'A'
        elif composite >= 0.85:
            return 'A-'
        elif composite >= 0.80:
            return 'B+'
        elif composite >= 0.75:
            return 'B'
        elif composite >= 0.70:
            return 'B-'
        elif composite >= 0.65:
            return 'C+'
        elif composite >= 0.60:
            return 'C'
        else:
            return 'D'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_extraction(extraction: Dict) -> ValidationResult:
    """Quick validation of single extraction."""
    validator = ExtractionValidator()
    return validator.validate_single(extraction)


def validate_batch(extractions: List[Dict]) -> Tuple[List[ValidationResult], Dict]:
    """
    Validate a batch of extractions.
    
    Returns:
        Tuple of (validation_results, quality_report)
    """
    validator = ExtractionValidator()
    
    # First pass: individual validation
    results = [validator.validate_single(e) for e in extractions]
    
    # Compute batch stats for anomaly detection
    validator.compute_batch_stats(extractions)
    
    # Second pass: anomaly detection
    for extraction, result in zip(extractions, results):
        anomalies = validator.detect_anomalies(extraction)
        for anomaly in anomalies:
            result.add_issue(anomaly, 'anomaly')
    
    # Generate report
    report = validator.get_quality_report(results)
    
    return results, report
