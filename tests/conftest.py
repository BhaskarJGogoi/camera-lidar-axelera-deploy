import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return PROJECT_ROOT / "data" / "kitti_demo_clip"


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    return PROJECT_ROOT / "weights"


@pytest.fixture(scope="session")
def seq() -> str:
    return "0011"
