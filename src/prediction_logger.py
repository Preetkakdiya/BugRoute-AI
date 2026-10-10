"""
BugRoute AI — Safe Prediction Correction Audit Logger Utility
Handles safe persistence, atomic writes, cross-process file locking,
data sanitization, corrupted JSON file preservation, and recovery.
"""

import os
import sys
import json
import time
import re
import uuid
import logging
import tempfile
import threading
from typing import Dict, List, Any, Optional

try:
    from filelock import FileLock, Timeout
except ImportError:
    class FileLock:
        def __init__(self, lock_file: str, timeout: float = 10.0):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    Timeout = Exception

logger = logging.getLogger("prediction_logger")
_THREAD_LOCK = threading.Lock()

SENSITIVE_PATTERNS = [
    (re.compile(r'(?i)(password|passwd|secret|token|api_key|apikey|auth_key)\s*[:=]\s*["\']?([^"\'\s&]+)["\']?'), r'\1=[REDACTED]'),
    (re.compile(r'(?i)bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*'), 'Bearer [REDACTED_TOKEN]'),
    (re.compile(r'(?i)ey[Jj][a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+'), '[REDACTED_JWT]')
]


def sanitize_text(text: Optional[str]) -> str:
    """Strips whitespace and redacts sensitive credentials (passwords, JWTs, API tokens)."""
    if not text or not isinstance(text, str):
        return ""
    text_clean = text.strip()
    for pattern, replacement in SENSITIVE_PATTERNS:
        text_clean = pattern.sub(replacement, text_clean)
    return text_clean


def sanitize_correction_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and sanitizes a prediction correction dictionary entry."""
    if not isinstance(entry, dict):
        raise ValueError("Correction entry must be a dictionary.")

    title = sanitize_text(str(entry.get("title", "")))
    desc = sanitize_text(str(entry.get("description", "")))
    feedback = sanitize_text(str(entry.get("user_feedback", "")))

    if not title and not desc:
        raise ValueError("Correction entry must include a non-empty title or description.")

    sanitized = {
        "timestamp": entry.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "description": desc,
        "model_used": sanitize_text(str(entry.get("model_used", "Linear SVM"))),
        "predicted_class": sanitize_text(str(entry.get("predicted_class", "Unknown"))),
        "predicted_confidence": float(entry.get("predicted_confidence", 0.0)),
        "user_corrected_class": sanitize_text(str(entry.get("user_corrected_class", "Other"))),
        "user_feedback": feedback
    }
    return sanitized


def generate_backup_filename(base_path: str, reason: str) -> str:
    """Generates a collision-proof backup filename with timestamp and random UUID snippet."""
    dir_name, file_name = os.path.split(base_path)
    now_str = time.strftime("%Y%m%d_%H%M%S")
    sub_second = int((time.time() % 1) * 1000000)
    rand_id = uuid.uuid4().hex[:8]
    backup_name = f"{file_name}.{reason}_{now_str}_{sub_second:06d}_{rand_id}"
    return os.path.join(dir_name, backup_name)


def backup_problematic_file(file_path: str, reason: str) -> Optional[str]:
    """Preserves problematic/corrupted file in a collision-proof backup file."""
    if not os.path.exists(file_path):
        return None
    backup_path = generate_backup_filename(file_path, reason)
    try:
        os.rename(file_path, backup_path)
        logger.warning(f"Preserved problematic file '{file_path}' to backup '{backup_path}'. Reason: {reason}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup file '{file_path}' to '{backup_path}': {e}")
        return None


def save_prediction_correction(correction_entry: dict, reports_dir: str = "reports", lock_timeout: float = 5.0) -> bool:
    """
    Saves a user prediction correction entry to user_prediction_corrections.json.
    - Uses thread and process file-locking to prevent concurrent write data loss.
    - Performs atomic file replacement via temporary files.
    - Preserves corrupted JSON and non-list JSON formats into collision-proof backups.
    - Aborts write operation if backup rename fails to prevent accidental data loss.
    - Sanitizes sensitive input information.

    Note: File-locking handles single-node multi-thread and multi-process concurrency.
          For distributed multi-server scale, a transactional database is recommended.
    """
    try:
        os.makedirs(reports_dir, exist_ok=True)
    except Exception as dir_err:
        logger.error(f"Cannot create or access reports directory '{reports_dir}': {dir_err}")
        return False

    corr_file = os.path.abspath(os.path.join(reports_dir, "user_prediction_corrections.json"))
    lock_file = corr_file + ".lock"

    try:
        sanitized_entry = sanitize_correction_entry(correction_entry)
    except Exception as ve:
        logger.error(f"Validation failed for correction entry: {ve}")
        return False

    with _THREAD_LOCK:
        try:
            with FileLock(lock_file, timeout=lock_timeout):
                existing_corr = []
                if os.path.exists(corr_file):
                    file_parsed = False
                    data = None
                    try:
                        with open(corr_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        file_parsed = True
                    except (json.JSONDecodeError, Exception) as parse_err:
                        logger.warning(f"Failed to parse JSON from '{corr_file}': {parse_err}. Preserving corrupted file.")
                        b_res = backup_problematic_file(corr_file, "corrupted")
                        if b_res is None:
                            logger.error(f"Failed to backup corrupted file '{corr_file}'. Aborting write to prevent data loss.")
                            return False
                        existing_corr = []

                    if file_parsed:
                        if isinstance(data, list):
                            existing_corr = data
                        else:
                            logger.warning(f"File '{corr_file}' root is not a list. Preserving invalid structure file.")
                            b_res = backup_problematic_file(corr_file, "invalid_struct")
                            if b_res is None:
                                logger.error(f"Failed to backup invalid structure file '{corr_file}'. Aborting write to prevent data loss.")
                                return False
                            existing_corr = []

                existing_corr.append(sanitized_entry)

                # Atomic write via temp file in target directory
                try:
                    tmp_fd, tmp_path = tempfile.mkstemp(prefix="corr_", suffix=".tmp", dir=reports_dir)
                except Exception as tmp_err:
                    logger.error(f"Failed to create temp file in '{reports_dir}': {tmp_err}")
                    return False

                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f_tmp:
                        json.dump(existing_corr, f_tmp, indent=2)
                        f_tmp.flush()
                        os.fsync(f_tmp.fileno())
                    
                    os.replace(tmp_path, corr_file)
                    return True
                except Exception as write_err:
                    logger.error(f"Atomic write failed to '{corr_file}': {write_err}")
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                    return False

        except Timeout:
            logger.error(f"Timeout waiting for lock on '{lock_file}'.")
            return False
        except Exception as ex:
            logger.error(f"Unexpected error saving prediction correction: {ex}")
            return False
