"""Build the PyO3 extension and drop it into the `autobm25` package.

    python rust/build_pyext.py            # release build (default)
    python rust/build_pyext.py --debug    # faster to compile, slower to run
    python rust/build_pyext.py --check    # build, then verify it imports

The extension is `autobm25._autobm25_rs`; without it the package quietly uses its
pure-Python implementation, so this step is optional.

Toolchain notes
---------------
* On Windows the MSVC toolchain is the usual choice (`rustup default stable-msvc`).
  When only MinGW is installed, pass `--toolchain stable-x86_64-pc-windows-gnu`;
  the resulting `abi3` `.pyd` loads in CPython just like an MSVC one.
* `abi3-py39` means the built module works on CPython 3.9 and newer, matching the
  package's `requires-python`.
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACKAGE_DIR = os.path.join(ROOT, "autobm25")
MODULE_NAME = "_autobm25_rs"


def mingw_toolchains():
    """Installed `*-windows-gnu` toolchains, in rustup's order.

    A `stable-x86_64-pc-windows-msvc` default that cannot find `link.exe` (no
    Visual C++ build tools installed) fails at link time, while the MinGW
    toolchain next to it works - so the builder retries rather than guessing from
    `rustc --version`, which succeeds either way.
    """
    try:
        proc = subprocess.run(
            ["rustup", "toolchain", "list"], capture_output=True, text=True
        )
    except (FileNotFoundError, OSError):
        return []
    if proc.returncode != 0:
        return []
    return [
        line.split()[0]
        for line in proc.stdout.splitlines()
        if "windows-gnu" in line and line.split()
    ]


def toolchain_candidates(explicit=None):
    """Toolchains to try in order: the explicit one, else default then MinGW."""
    if explicit:
        return [explicit]
    return [None] + mingw_toolchains()


def built_library(target_dir):
    """Path of the cdylib cargo just produced (name differs per platform)."""
    names = ["autobm25.dll"] if os.name == "nt" else ["libautobm25.so", "libautobm25.dylib"]
    for name in names:
        candidate = os.path.join(target_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(f"cargo did not produce a shared library in {target_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build the debug profile")
    parser.add_argument("--toolchain", default=None,
                        help="rustup toolchain, e.g. stable-x86_64-pc-windows-gnu")
    parser.add_argument("--check", action="store_true",
                        help="import the built module afterwards")
    args = parser.parse_args()

    profile = "debug" if args.debug else "release"

    env = dict(os.environ)
    # PyO3 needs to know which interpreter to build against; the running one is
    # almost always the right answer.
    env.setdefault("PYO3_PYTHON", sys.executable)
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")

    candidates = toolchain_candidates(args.toolchain)
    failure = None
    for position, toolchain in enumerate(candidates):
        cmd = ["cargo"]
        if toolchain:
            cmd.append(f"+{toolchain}")
        cmd += ["build", f"--{profile}", "--features", "python"]
        print(f"[build_pyext] {' '.join(cmd)}  (cwd={HERE})")
        try:
            proc = subprocess.run(cmd, cwd=HERE, env=env)
        except FileNotFoundError:
            raise SystemExit("cargo not found on PATH; install Rust from rustup.rs")
        if proc.returncode == 0:
            break
        failure = toolchain or "the default toolchain"
        remaining = candidates[position + 1:]
        if remaining:
            print(
                f"[build_pyext] {failure} failed to build; trying "
                f"{', '.join(t for t in remaining if t) or 'the next toolchain'}"
            )
    else:
        raise SystemExit(f"cargo build failed for {failure}")

    target_dir = os.path.join(HERE, "target", profile)
    library = built_library(target_dir)
    destination = os.path.join(PACKAGE_DIR, f"{MODULE_NAME}.pyd"
                               if os.name == "nt" else f"{MODULE_NAME}.so")
    shutil.copyfile(library, destination)
    print(f"[build_pyext] {library} -> {destination}")

    if args.check:
        sys.path.insert(0, ROOT)
        from autobm25 import _accel  # noqa: E402

        print(f"[build_pyext] backend={_accel.BACKEND} info={_accel.backend_info()}")
        if not _accel.ENABLED:
            raise SystemExit("built the extension but it did not import")


if __name__ == "__main__":
    main()
