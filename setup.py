"""setuptools entry point (project metadata lives in `pyproject.toml`).

This file exists for one reason: to make the optional Rust extension part of a
normal `pip install .` / `pip wheel .`, so nobody has to run a separate step.

Three cases, checked in order before the build starts:

1. a compiled `autobm25/_autobm25_rs.<ext>` for this platform is already in the
   tree (a wheel built here, or a checkout carrying a prebuilt binary) - it is
   used as is;
2. otherwise, if `cargo` is on PATH, `rust/build_pyext.py` compiles it;
3. otherwise the package installs pure Python, which is fully functional and
   only slower.

Set ``AUTOBM25_SKIP_RUST_BUILD=1`` to skip step 2.

Nothing here is required for correctness: `autobm25/_accel.py` falls back to the
Python engine whenever the compiled module is absent, so **an end user never
needs a Rust toolchain** - they install a wheel that already contains the
compiled extension. See the "Shipping it to other users" section of the README.
"""

import os
import shutil
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:  # setuptools >= 70.1 ships its own bdist_wheel
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(HERE, "autobm25")
MODULE_NAME = "_autobm25_rs"
BUILD_SCRIPT = os.path.join(HERE, "rust", "build_pyext.py")

# Commands that should trigger the Rust build; anything else (metadata queries,
# `python setup.py --version`, sdist) stays cheap.
_BUILD_COMMANDS = (
    "bdist_wheel",
    "build",
    "build_py",
    "install",
    "develop",
    "editable_wheel",
)


def extension_suffix():
    """Suffix a compiled Python extension gets on this platform."""
    if sys.platform == "win32":
        return ".pyd"
    return ".so"


def extension_path():
    return os.path.join(PACKAGE_DIR, MODULE_NAME + extension_suffix())


def skipped_by_env():
    return os.environ.get("AUTOBM25_SKIP_RUST_BUILD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def build_with_cargo():
    """Compile the extension in place; returns True when it appeared."""
    if skipped_by_env():
        print("[autobm25] AUTOBM25_SKIP_RUST_BUILD is set - installing pure Python")
        return False
    if shutil.which("cargo") is None:
        print(
            "[autobm25] cargo not found - installing the pure-Python package.\n"
            "           `pip install autobm25` wheels already contain the compiled\n"
            "           extension; see the README for the other options."
        )
        return False
    print(f"[autobm25] building the Rust extension: {sys.executable} {BUILD_SCRIPT}")
    try:
        subprocess.run([sys.executable, BUILD_SCRIPT], cwd=HERE, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[autobm25] Rust build failed ({exc}); falling back to pure Python")
        return False
    return os.path.exists(extension_path())


def resolve_extension():
    """The compiled extension to ship, or None for a pure-Python build."""
    if os.path.exists(extension_path()):
        print(f"[autobm25] using prebuilt {extension_path()}")
        return extension_path()
    if any(command in sys.argv[1:] for command in _BUILD_COMMANDS):
        if build_with_cargo():
            return extension_path()
    return None


# Resolved *before* setup() runs: the wheel tag has to reflect reality, so we must
# know whether a platform binary is going out before the distribution is created.
ARTIFACT = resolve_extension()


class BinaryDistribution(Distribution):
    """Marks the distribution non-pure when the compiled extension ships.

    Without this, setuptools would tag a wheel containing a compiled `.pyd` as
    `py3-none-any`, which pip would happily install on a platform it cannot run on.
    """

    def has_ext_modules(self):
        return ARTIFACT is not None


if _bdist_wheel is not None:

    class PlatformWheel(_bdist_wheel):
        """Tag the wheel `cp39-abi3-<platform>` for the `abi3-py39` build."""

        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = ARTIFACT is None

        def get_tag(self):
            python, abi, plat = super().get_tag()
            if ARTIFACT is not None:
                # PyO3 is configured with `abi3-py39`: one binary, CPython 3.9+.
                return "cp39", "abi3", plat
            return python, abi, plat

else:  # pragma: no cover - bdist_wheel unavailable
    PlatformWheel = None


class RustAwareBuildPy(build_py):
    """Keep a stale compiled extension out of a pure-Python wheel.

    `build_py` copies files into `build/lib` but never deletes from it, so a build
    that once shipped the extension leaves `build/lib/autobm25/_autobm25_rs.pyd`
    behind. Without this, the next pure-Python build would silently package that
    leftover binary into a `py3-none-any` wheel.
    """

    def run(self):
        super().run()
        if ARTIFACT is not None:
            return
        package_dir = os.path.join(self.build_lib, "autobm25")
        for suffix in (".pyd", ".so", ".dylib"):
            stale = os.path.join(package_dir, MODULE_NAME + suffix)
            if os.path.exists(stale):
                print(f"[autobm25] removing stale {stale} from the build tree")
                os.remove(stale)


setup(
    distclass=BinaryDistribution,
    cmdclass={
        "build_py": RustAwareBuildPy,
        **({} if PlatformWheel is None else {"bdist_wheel": PlatformWheel}),
    },
)
