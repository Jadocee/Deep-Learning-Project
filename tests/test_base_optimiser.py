from os.path import join
from pathlib import Path
from unittest import TestCase

from utils.definitions import STUDIES_DIR


class TestBaseOptimiser(TestCase):

    def test_save_results(self):
        output_dir: str = join(STUDIES_DIR, "test_study")
        output_path: Path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.assertTrue(output_path.exists())
