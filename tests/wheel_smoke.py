"""Install a built wheel outside the checkout and verify both backends."""

import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROBE = r"""
import json
import sys
import autobm25

extension = sys.modules.get("autobm25._autobm25_rs")
retriever = autobm25.AutoBM25().index(
    ["the quick brown fox", "quick quick fox jumps", "a lazy dog sleeps"]
)
print(json.dumps({
    "backend": autobm25.BACKEND,
    "extension": extension.__file__ if extension else None,
    "hits": [doc_id for doc_id, _ in retriever.search("quick fox", top_k=2)],
}))
"""


def _wheel_path(argument):
    path = Path(argument)
    if path.is_dir():
        wheels = sorted(glob.glob(str(path / "*.whl")))
        if len(wheels) != 1:
            raise SystemExit(f"expected exactly one wheel in {path}, found {wheels}")
        return Path(wheels[0])
    if not path.is_file():
        raise SystemExit(f"wheel not found: {path}")
    return path


def _probe(workdir, target, pure_python=False):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    if pure_python:
        env["AUTOBM25_NO_RUST"] = "1"
    else:
        env.pop("AUTOBM25_NO_RUST", None)
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("wheel probe failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    wheel = _wheel_path(sys.argv[1] if len(sys.argv) > 1 else ".")
    workdir = Path(tempfile.mkdtemp(prefix="autobm25-wheel-smoke-"))
    try:
        target = workdir / "site"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )

        rust = _probe(workdir, target)
        pure = _probe(workdir, target, pure_python=True)

        print(f"wheel: {wheel.name}")
        print(f"rust backend: {rust}")
        print(f"python backend: {pure}")
        if rust["backend"] != "rust":
            raise SystemExit("wheel did not load the Rust extension")
        extension = Path(rust["extension"]).resolve() if rust["extension"] else None
        if not extension or not extension.is_relative_to(target.resolve()):
            raise SystemExit("Rust extension did not come from the installed wheel")
        if pure["backend"] != "python":
            raise SystemExit("AUTOBM25_NO_RUST did not select the Python backend")
        if rust["hits"] != pure["hits"]:
            raise SystemExit("Rust and Python backends returned different results")
        print("wheel smoke test: PASS")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
