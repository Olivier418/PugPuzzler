"""Run the whole test suite: just run this file (no terminal arguments needed)."""
import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
