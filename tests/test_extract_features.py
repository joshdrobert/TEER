from __future__ import annotations

import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_features.py"
SPEC = importlib.util.spec_from_file_location("extract_features", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FeatureTableTest(unittest.TestCase):
    def test_bundled_geometry_feature_table_has_all_105_subjects(self) -> None:
        valve_dir = Path(__file__).resolve().parents[1] / "valves"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.csv"
            rows = MODULE.extract_table(valve_dir, output)

            self.assertEqual(len(rows), 105)
            self.assertEqual(
                [row["subject_id"] for row in rows],
                [f"train_{i:03d}" for i in range(1, 106)],
            )
            self.assertEqual(len({row["subject_id"] for row in rows}), 105)
            for row in rows:
                for key, value in row.items():
                    if key != "subject_id" and not isinstance(value, bool):
                        self.assertTrue(math.isfinite(float(value)), (row["subject_id"], key, value))

            with output.open(newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(len(written), 105)
            self.assertEqual(list(written[0]), MODULE.FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
