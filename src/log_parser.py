"""
BugRoute AI - Multi-Issue Log Parser
Extracts individual error events and multiline stack traces from log files.
"""

import re
from typing import List, Dict, Any


TIMESTAMP_PREFIX_PATTERN = re.compile(
    r'^(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\[\d{4}-\d{2}-\d{2}|\[\d{2}:\d{2}:\d{2})'
)

# Match explicit log level tokens e.g. [ERROR], ERROR:, [CRITICAL], INFO:
SEVERITY_PATTERN = re.compile(
    r'(?:^|[\s\[])(ERROR|CRITICAL|FATAL|SEVERE|EMERGENCY|WARN|WARNING|INFO|DEBUG|TRACE|NOTICE)(?:[:\]\s]|$)'
)

ERROR_SEVERITY_PATTERN = re.compile(
    r'(?:^|[\s\[])(ERROR|CRITICAL|FATAL|SEVERE|EMERGENCY)(?:[:\]\s]|$)'
)

EXCEPTION_PATTERN = re.compile(
    r'\b(?:[a-zA-Z0-9_.]+\.)*(?:[a-zA-Z0-9_]+(?:Exception|Error|Fault|Failure|Timeout|Deadlock|Crash|OOM))\b',
    re.IGNORECASE
)

PYTHON_TRACEBACK_PATTERN = re.compile(
    r'Traceback \(most recent call last\):',
    re.IGNORECASE
)


def extract_log_issues(log_content: str) -> List[Dict[str, Any]]:
    """
    Parses full log text content and extracts distinct error issues, 
    grouping multiline stack traces with their respective error headers.
    """
    if not log_content or not isinstance(log_content, str):
        return []

    lines = log_content.splitlines()
    if not lines:
        return []

    issues = []
    current_issue_lines = []
    current_line_start = 0
    current_title = ""
    current_level = "ERROR"

    def finalize_current_issue():
        nonlocal current_issue_lines, current_line_start, current_title, current_level
        if current_issue_lines:
            full_text = "\n".join(current_issue_lines).strip()
            if len(full_text) >= 10:
                if not current_title:
                    current_title = current_issue_lines[0].strip()[:120]
                issues.append({
                    "issue_number": len(issues) + 1,
                    "line_number": current_line_start + 1,
                    "title": current_title,
                    "description": full_text,
                    "log_level": current_level
                })
        current_issue_lines = []
        current_title = ""
        current_level = "ERROR"

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current_issue_lines:
                if idx + 1 < len(lines):
                    next_stripped = lines[idx + 1].strip()
                    if next_stripped.startswith('at ') or next_stripped.startswith('File ') or next_stripped.startswith('line '):
                        current_issue_lines.append(line)
                        continue
                finalize_current_issue()
            continue

        is_indented = line.startswith(' ') or line.startswith('\t')
        has_timestamp = bool(TIMESTAMP_PREFIX_PATTERN.match(stripped))
        has_severity = bool(SEVERITY_PATTERN.search(line))
        has_error_severity = bool(ERROR_SEVERITY_PATTERN.search(line))

        # A line is a fresh log entry header if it has a timestamp or an unindented severity marker
        is_new_log_entry = has_timestamp or (has_severity and not is_indented)

        is_error_trigger = (
            has_error_severity or 
            bool(PYTHON_TRACEBACK_PATTERN.search(line)) or 
            (bool(EXCEPTION_PATTERN.search(line)) and not is_indented)
        )

        if is_new_log_entry:
            # New log entry header encountered
            if has_error_severity or (has_timestamp and is_error_trigger):
                finalize_current_issue()
                current_line_start = idx
                current_title = stripped[:120]
                current_level = "CRITICAL" if ("CRITICAL" in line or "FATAL" in line) else "ERROR"
                current_issue_lines.append(line)
            else:
                # Normal log line (e.g. INFO, DEBUG)
                finalize_current_issue()

        else:
            # Not a timestamped / severity log entry header
            if current_issue_lines:
                # Append continuation lines to active issue
                current_issue_lines.append(line)
            else:
                # No issue active, check if this unprefixed line starts an error issue
                if is_error_trigger:
                    current_line_start = idx
                    current_title = stripped[:120]
                    current_level = "CRITICAL" if ("CRITICAL" in line or "FATAL" in line) else "ERROR"
                    current_issue_lines.append(line)

    finalize_current_issue()
    return issues
