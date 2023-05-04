from os.path import dirname, abspath

ROOT: str = dirname(dirname(dirname(abspath(__file__))))
"""
The root directory of the project.
"""

DATA_DIR: str = f"{ROOT}/data"
"""
The directory where the datasets are stored.
"""

OUT_DIR: str = f"{ROOT}/out"
"""
The directory where the output files are stored.
"""
