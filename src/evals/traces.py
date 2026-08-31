"""
SQLite trace logging for observability and feedback collection.

Why traces?
- No gold labels (we can't label 1000 articles manually)
- Traces capture latency, cache hits, label confidence for later analysis
- User feedback (thumbs up/down) becomes a quality signal
- Enables iterative improvement: run evals, see what's working, adjust

Trace schema:
  request_id, timestamp, query, ticker, retrieved_ids (JSON),
  event_types (JSON), risk_levels (JSON), brief_confidence,
  latency_ms, cache_hits, user_feedback, notes
"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class TraceStore:
    """SQLite trace database for evals."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize trace store.

        Args:
            db_path: Path to SQLite DB. If None, uses Config.TRACE_DB.
        """
        self.db_path = db_path or Config.TRACE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create trace table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    ticker TEXT,

                    retrieved_count INTEGER,
                    retrieved_ids TEXT,  -- JSON: ["id1", "id2", ...]
                    event_types TEXT,    -- JSON: ["earnings", "geopolitics"]
                    risk_levels TEXT,    -- JSON: ["low", "high"]
                    avg_risk_score REAL,

                    brief_confidence REAL,
                    brief_summary TEXT,

                    latency_ms REAL,
                    cache_hits INTEGER,

                    user_feedback TEXT,  -- "thumbs_up", "thumbs_down", "neutral"
                    user_notes TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def log_trace(
        self,
        request_id: str,
        query: str,
        ticker: Optional[str],
        retrieved_ids: list[str],
        event_types: list[str],
        risk_levels: list[str],
        avg_risk_score: float,
        brief_confidence: float,
        brief_summary: str,
        latency_ms: float,
        cache_hits: int = 0,
    ) -> bool:
        """
        Log a trace after running the full L5 pipeline.

        Args:
            request_id: Unique request identifier
            query: User query
            ticker: Stock ticker (if extracted)
            retrieved_ids: List of article IDs returned by retriever
            event_types: List of unique event types in retrieved articles
            risk_levels: List of unique risk levels
            avg_risk_score: Average risk score across articles
            brief_confidence: Confidence score of the brief (0-1)
            brief_summary: 1-2 line summary of the brief
            latency_ms: Total pipeline latency in milliseconds
            cache_hits: Number of cache hits during pipeline

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO traces (
                        request_id, query, ticker,
                        retrieved_count, retrieved_ids, event_types, risk_levels, avg_risk_score,
                        brief_confidence, brief_summary,
                        latency_ms, cache_hits,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        query,
                        ticker,
                        len(retrieved_ids),
                        json.dumps(retrieved_ids),
                        json.dumps(event_types),
                        json.dumps(risk_levels),
                        avg_risk_score,
                        brief_confidence,
                        brief_summary,
                        latency_ms,
                        cache_hits,
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
                logger.info(f"✓ Trace logged: {request_id}")
                return True

        except sqlite3.IntegrityError as e:
            logger.warning(f"Trace already exists: {request_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to log trace: {e}")
            return False

    def log_feedback(
        self,
        request_id: str,
        user_feedback: str,
        user_notes: Optional[str] = None,
    ) -> bool:
        """
        Log user feedback on a brief (thumbs up/down).

        Args:
            request_id: Request ID of the brief
            user_feedback: "thumbs_up", "thumbs_down", or "neutral"
            user_notes: Optional notes from the user

        Returns:
            True if updated successfully
        """
        if user_feedback not in ["thumbs_up", "thumbs_down", "neutral"]:
            logger.warning(f"Invalid feedback: {user_feedback}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE traces
                    SET user_feedback = ?, user_notes = ?
                    WHERE request_id = ?
                    """,
                    (user_feedback, user_notes, request_id),
                )
                conn.commit()
                logger.info(f"✓ Feedback logged: {request_id} = {user_feedback}")
                return True

        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            return False

    def get_trace(self, request_id: str) -> Optional[dict]:
        """
        Retrieve a trace by request ID.

        Args:
            request_id: Request ID

        Returns:
            Trace dict, or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM traces WHERE request_id = ?",
                    (request_id,),
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)

        except Exception as e:
            logger.error(f"Failed to retrieve trace: {e}")

        return None

    def get_all_traces(self, limit: int = 100) -> list[dict]:
        """
        Retrieve all traces (for metrics computation).

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of trace dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to retrieve traces: {e}")
            return []

    def get_feedback_summary(self) -> dict:
        """
        Summarize user feedback from all traces.

        Returns:
            Dict with keys: thumbs_up_count, thumbs_down_count, neutral_count, total_feedback
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN user_feedback = 'thumbs_up' THEN 1 ELSE 0 END) as up_count,
                        SUM(CASE WHEN user_feedback = 'thumbs_down' THEN 1 ELSE 0 END) as down_count,
                        SUM(CASE WHEN user_feedback = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                    FROM traces
                    WHERE user_feedback IS NOT NULL
                    """
                )
                row = cursor.fetchone()
                if row:
                    up, down, neutral = row
                    total = (up or 0) + (down or 0) + (neutral or 0)
                    return {
                        "thumbs_up": up or 0,
                        "thumbs_down": down or 0,
                        "neutral": neutral or 0,
                        "total": total,
                    }

        except Exception as e:
            logger.error(f"Failed to compute feedback summary: {e}")

        return {"thumbs_up": 0, "thumbs_down": 0, "neutral": 0, "total": 0}
