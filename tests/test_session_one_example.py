"""Regression test for Session 1's second ratings example."""
import os
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent


class SessionOneExampleTest(unittest.TestCase):
    def test_second_table_prints_neighbour_and_recommendation(self):
        path = ROOT / "sessions" / "session-01" / "lesson.py"
        res = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True,
            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8"),
            cwd=ROOT, timeout=180,
        )
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("Second invented ratings table (5 people x 6 films)", res.stdout)
        self.assertIn("nearest person to Mina: Noah", res.stdout)
        self.assertIn("recommendation for Mina: Night Chase", res.stdout)


if __name__ == "__main__":
    unittest.main()
