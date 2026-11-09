import re
from pathlib import Path

import autobm25


ROOT = Path(__file__).resolve().parents[1]


def test_all_release_versions_are_in_sync():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (ROOT / "rust" / "Cargo.toml").read_text(encoding="utf-8")
    lock = (ROOT / "rust" / "Cargo.lock").read_text(encoding="utf-8")

    project_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    cargo_version = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE).group(1)
    lock_version = re.search(
        r'\[\[package\]\]\nname = "autobm25"\nversion = "([^"]+)"',
        lock,
    ).group(1)

    assert autobm25.__version__ == project_version == cargo_version == lock_version
    if autobm25.BACKEND == "rust":
        assert autobm25.backend_info()["version"] == project_version
