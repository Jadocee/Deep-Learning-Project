from os.path import dirname, abspath, join

ROOT: str = dirname(dirname(dirname(abspath(__file__))))
"""
The root directory of the project.
"""

DATA_DIR: str = join(ROOT, "data")
"""
The directory where the datasets are stored.
"""

OUT_DIR: str = join(ROOT, "out")
"""
The directory where the output files are stored.
"""

MODELS_DIR: str = join(OUT_DIR, "models")
"""
The directory for storing the trained models.
"""

STUDIES_DIR: str = join(OUT_DIR, "studies")
"""
The directory for storing the Optuna studies.
"""
