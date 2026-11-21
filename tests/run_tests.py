"""Run every test: python -I -B tests/run_tests.py

The last line is RESULT pass=N fail=M skip=K. Any failure exits 1.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
sys.dont_write_bytecode = True


def main():
    suite = unittest.defaultTestLoader.discover(HERE, pattern="test_*.py", top_level_dir=HERE)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(f"RESULT pass={result.testsRun - failed - len(result.skipped)} fail={failed} skip={len(result.skipped)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
