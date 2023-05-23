from os.path import dirname, abspath, join
from typing import Final

ROOT: Final[str] = dirname(dirname(dirname(abspath(__file__))))
"""
The root directory of the project.
"""

DATA_DIR: Final[str] = join(ROOT, "data")
"""
The directory where the datasets are stored.
"""

OUT_DIR: Final[str] = join(ROOT, "out")
"""
The directory where the output files are stored.
"""

MODELS_DIR: Final[str] = join(OUT_DIR, "models")
"""
The directory for storing the trained models.
"""

STUDIES_DIR: Final[str] = join(OUT_DIR, "studies")
"""
The directory for storing the Optuna studies.
"""

VOCABS_DIR: Final[str] = join(OUT_DIR, "vocabs")
"""
The directory for storing the vocabulary files.
"""

TRAINED_DIR: str = join(ROOT, "trained_models")
"""
The directory for the trained cnn models.
"""
