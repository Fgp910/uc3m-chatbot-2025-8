"""
Checkpoint Module - Resume Capability for Production Extraction

Features:
1. Save progress after each document
2. Resume from last checkpoint on crash
3. Track success/failure per document
4. Estimate time remaining

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """State saved at each checkpoint."""

    # Progress tracking
    total_documents: int = 0
    processed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0

    # Document tracking
    processed_files: List[str] = None  # Filenames already processed
    failed_files: List[str] = None     # Filenames that failed

    # Timing
    start_time: str = ""
    last_checkpoint: str = ""
    avg_seconds_per_doc: float = 0.0

    # Configuration (for validation on resume)
    config_hash: str = ""

    def __post_init__(self):
        if self.processed_files is None:
            self.processed_files = []
        if self.failed_files is None:
            self.failed_files = []


class CheckpointManager:
    """
    Manage extraction checkpoints for resume capability.

    Usage:
        manager = CheckpointManager("./output/checkpoint.json")

        # Check if resuming
        if manager.has_checkpoint():
            state = manager.load()
            skip_files = set(state.processed_files)

        # During processing
        for pdf in pdfs:
            if pdf.name in skip_files:
                continue

            result = extract(pdf)
            manager.mark_processed(pdf.name, success=True)
            manager.save()  # Save after each document

        # Clean up on completion
        manager.clear()
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.state = CheckpointState()
        self._processing_start = None

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_path.exists()

    def load(self) -> CheckpointState:
        """Load checkpoint from disk."""
        if not self.has_checkpoint():
            raise FileNotFoundError(f"No checkpoint at {self.checkpoint_path}")

        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)

        self.state = CheckpointState(
            total_documents=data.get('total_documents', 0),
            processed_count=data.get('processed_count', 0),
            successful_count=data.get('successful_count', 0),
            failed_count=data.get('failed_count', 0),
            processed_files=data.get('processed_files', []),
            failed_files=data.get('failed_files', []),
            start_time=data.get('start_time', ''),
            last_checkpoint=data.get('last_checkpoint', ''),
            avg_seconds_per_doc=data.get('avg_seconds_per_doc', 0.0),
            config_hash=data.get('config_hash', ''),
        )

        logger.info(f"✓ Loaded checkpoint: {self.state.processed_count}/{self.state.total_documents} documents")

        return self.state

    def save(self):
        """Save checkpoint to disk."""
        self.state.last_checkpoint = datetime.now().isoformat()

        # Ensure directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_path, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

        logger.debug(f"Checkpoint saved: {self.state.processed_count} processed")

    def initialize(self, total_documents: int, config_hash: str = ""):
        """Initialize a new checkpoint."""
        self.state = CheckpointState(
            total_documents=total_documents,
            start_time=datetime.now().isoformat(),
            config_hash=config_hash,
        )
        self._processing_start = datetime.now()
        self.save()

        logger.info(f"Initialized checkpoint for {total_documents} documents")

    def mark_processed(self, filename: str, success: bool, error_msg: str = ""):
        """Mark a document as processed."""

        # Update timing
        if self._processing_start is None:
            self._processing_start = datetime.now()

        self.state.processed_count += 1

        if success:
            self.state.successful_count += 1
            self.state.processed_files.append(filename)
        else:
            self.state.failed_count += 1
            self.state.failed_files.append(filename)

        # Update average time
        elapsed = (datetime.now() - self._processing_start).total_seconds()
        self.state.avg_seconds_per_doc = elapsed / self.state.processed_count

    def get_skip_set(self) -> Set[str]:
        """Get set of filenames to skip (already processed)."""
        return set(self.state.processed_files)

    def get_remaining_count(self) -> int:
        """Get number of documents remaining."""
        return self.state.total_documents - self.state.processed_count

    def get_eta(self) -> Optional[timedelta]:
        """Estimate time remaining."""
        if self.state.avg_seconds_per_doc <= 0:
            return None

        remaining = self.get_remaining_count()
        seconds = remaining * self.state.avg_seconds_per_doc

        return timedelta(seconds=seconds)

    def get_progress_str(self) -> str:
        """Get human-readable progress string."""
        pct = 100 * self.state.processed_count / self.state.total_documents if self.state.total_documents > 0 else 0
        eta = self.get_eta()
        eta_str = str(eta).split('.')[0] if eta else "unknown"

        return (
            f"Progress: {self.state.processed_count}/{self.state.total_documents} "
            f"({pct:.1f}%) | Success: {self.state.successful_count} | "
            f"Failed: {self.state.failed_count} | ETA: {eta_str}"
        )

    def clear(self):
        """Delete checkpoint file (call on successful completion)."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint cleared")

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total': self.state.total_documents,
            'processed': self.state.processed_count,
            'successful': self.state.successful_count,
            'failed': self.state.failed_count,
            'success_rate': round(100 * self.state.successful_count / self.state.processed_count, 1) if self.state.processed_count > 0 else 0,
            'avg_seconds_per_doc': round(self.state.avg_seconds_per_doc, 2),
            'failed_files': self.state.failed_files,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_config_hash(config: Dict) -> str:
    """
    Create hash of configuration for validation on resume.

    If config changes between runs, warn user.
    """
    import hashlib
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

"""
Example integration with SmartExtractor:

```python
def extract_batch_with_checkpoint(
    self,
    pdf_dir: Path,
    output_dir: Path,
    checkpoint_path: Path = None,
    limit: Optional[int] = None,
) -> List[ExtractionResult]:

    # Initialize checkpoint manager
    checkpoint_path = checkpoint_path or (output_dir / "checkpoint.json")
    checkpoint = CheckpointManager(checkpoint_path)

    # Find PDFs
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    # Check for existing checkpoint
    skip_files = set()
    if checkpoint.has_checkpoint():
        state = checkpoint.load()
        skip_files = checkpoint.get_skip_set()
        logger.info(f"Resuming from checkpoint: {len(skip_files)} already processed")
    else:
        checkpoint.initialize(len(pdfs))

    results = []

    for pdf_path in pdfs:
        # Skip if already processed
        if pdf_path.name in skip_files:
            logger.info(f"Skipping (already processed): {pdf_path.name}")
            continue

        try:
            result = self.extract(pdf_path)
            results.append(result)
            checkpoint.mark_processed(pdf_path.name, success=True)
        except Exception as e:
            logger.error(f"Error: {e}")
            checkpoint.mark_processed(pdf_path.name, success=False, error_msg=str(e))

        # Save checkpoint after each document
        checkpoint.save()

        # Log progress
        logger.info(checkpoint.get_progress_str())

    # Clear checkpoint on successful completion
    if checkpoint.get_remaining_count() == 0:
        checkpoint.clear()

    return results
```
"""
