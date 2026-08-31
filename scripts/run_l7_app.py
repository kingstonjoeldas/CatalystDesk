#!/usr/bin/env python
"""
Phase L7: Launch Streamlit App

This is a simple launcher script. In practice, you run:
  streamlit run app/streamlit_app.py

This script is here for completeness and to match the L0-L6 pattern.
"""

import subprocess
import sys
from pathlib import Path

# Add src to path (for imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


def main():
    """Launch Streamlit app."""
    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L7: Streamlit UI")
    logger.info("=" * 80)

    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"

    if not app_path.exists():
        logger.error(f"App not found: {app_path}")
        return 1

    logger.info(f"Launching Streamlit app: {app_path}")
    logger.info("Open http://localhost:8501 in your browser")
    logger.info("Press Ctrl+C to stop")

    try:
        # Run streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(Path(__file__).parent.parent),
        )
        return 0

    except KeyboardInterrupt:
        logger.info("App stopped by user")
        return 0

    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
