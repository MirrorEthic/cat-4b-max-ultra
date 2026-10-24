import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root: `import gateway`
pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    config.option.asyncio_mode = "auto"
