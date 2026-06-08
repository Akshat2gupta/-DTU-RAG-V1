"""
Root conftest: ensure dtu-chatbot/ is on sys.path so all packages import cleanly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
