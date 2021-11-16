from path import Path

SRC_DIR = Path(__file__).abspath().dirname()
ROOT_DIR = SRC_DIR.dirname()
VENV_DIR = ROOT_DIR / 'venv'
DATA_DIR = ROOT_DIR / 'data'
TEST_DIR = SRC_DIR / 'tests'
DEPS_DIR = ROOT_DIR / 'deps'