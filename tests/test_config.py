from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pathsnap.config import ROOT, data_dir


class DataPathTests(unittest.TestCase):
    def test_explicit_environment_path_has_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PATHSNAP_DATA_DIR": directory}):
                self.assertEqual(data_dir(), Path(directory))

    def test_repository_root_with_dataset_is_discovered(self):
        with patch.dict(os.environ):
            os.environ.pop("PATHSNAP_DATA_DIR", None)
            self.assertEqual(data_dir(), ROOT)
            self.assertEqual(len(list((data_dir() / "student_test").iterdir())), 10)


if __name__ == "__main__":
    unittest.main()
