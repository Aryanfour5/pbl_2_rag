"""
HybridBail: Utility Functions
Helper functions for the system
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import hashlib

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if needed
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def ensure_directory(path: str):
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)

def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
        return {}

def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save data to JSON file."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of file for change detection."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {filepath}: {e}")
        return ""

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_filename(filename: str) -> str:
    """Clean filename for display."""
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    name = ' '.join(name.split())
    return name

def validate_pdf(filepath: str) -> bool:
    """Validate if file is a valid PDF."""
    if not os.path.exists(filepath):
        return False
    
    if not filepath.lower().endswith('.pdf'):
        return False
    
    size = os.path.getsize(filepath)
    if size == 0:
        return False
    if size > 50 * 1024 * 1024:  # 50MB limit
        logger.warning(f"PDF too large: {filepath} ({size / 1024 / 1024:.1f}MB)")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
    except:
        return False
    
    return True

def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class ProgressTracker:
    """Track progress of long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta_seconds = (elapsed / self.current) * (self.total - self.current)
            eta = format_duration(eta_seconds)
        else:
            eta = "Unknown"
        
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta}", end='', flush=True)
    
    def finish(self):
        """Mark as finished."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{self.description}: Completed in {format_duration(elapsed)}")
