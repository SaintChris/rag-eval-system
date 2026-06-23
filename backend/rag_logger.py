"""
RAG System Logger — writes errors, rate limits, and learnings to a persistent log.
The agent reads this on startup to avoid repeating mistakes.
"""
import os
import json
import time
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag_system.log")


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_event(event_type: str, message: str, details: dict = None):
    """Append an event to the RAG system log."""
    _ensure_dir()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
        "details": details or {},
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_error(message: str, details: dict = None):
    log_event("ERROR", message, details)


def log_rate_limit(model: str, details: dict = None):
    log_event("RATE_LIMIT", f"Rate limited: {model}", details)


def log_learning(lesson: str, details: dict = None):
    log_event("LEARNING", lesson, details)


def log_query(query: str, model: str, success: bool, chars: int, time_s: float):
    log_event("QUERY", f"{'OK' if success else 'FAIL'}: {query[:60]}", {
        "model": model,
        "chars": chars,
        "time": time_s,
    })


def get_recent_errors(n: int = 10) -> list:
    """Get the last n error/learning entries."""
    _ensure_dir()
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry["type"] in ("ERROR", "LEARNING", "RATE_LIMIT"):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries[-n:]


def get_lessons() -> list:
    """Get all lessons learned."""
    _ensure_dir()
    if not os.path.exists(LOG_FILE):
        return []
    lessons = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry["type"] == "LEARNING":
                    lessons.append(entry)
            except json.JSONDecodeError:
                continue
    return lessons
