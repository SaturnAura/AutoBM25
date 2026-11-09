# Releasing AutoBM25

The version is declared in both `pyproject.toml` and `rust/Cargo.toml`; keep them
identical. Python bodies (`autobm25.__version__`) and the PyO3 module also expose
the same version.

## One-time PyPI setup

Use a PyPI project-scoped API token, not an account password. Store it only in the
`TWINE_PASSWORD` environment variable. For automated releases, configure PyPI
Trusted Publishing for repository `SaturnAura/AutoBM25`, workflow
`publish.yml`, and environment `pypi`; the workflow then needs no long-lived token.

## Manual release from a clean checkout

```bash
python -m pip install --upgrade build twine
python -m pytest -q
cargo test --manifest-path rust/Cargo.toml --locked
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` before the upload command.
Never commit a token, place it in `pyproject.toml`, or paste it into an issue.
If a token is exposed, revoke it in PyPI immediately and create a replacement.

## Automated release

1. Make sure CI is green on all three operating systems.
2. Create a signed Git tag such as `v0.1.1` and push it.
3. Publish a GitHub Release for that tag.
4. The publish workflow builds Linux, macOS, and Windows abi3 wheels plus an
   sdist, then uploads them through Trusted Publishing.

The Rust extension uses `abi3-py39`, so one wheel per platform/architecture serves
CPython 3.9 and newer. The sdist remains usable as a pure-Python package when no
Rust toolchain is installed.
