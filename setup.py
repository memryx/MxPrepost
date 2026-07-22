import os
import platform
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ROOT = Path(__file__).resolve().parent

if sys.platform.startswith("linux"):
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-fno-math-errno",
        "-funsafe-math-optimizations",
        "-ftree-vectorize",
        "-ftree-loop-ivcanon",
        "-ftree-loop-im",
        "-ffinite-math-only",
        "-fno-signed-zeros",
        "-fno-trapping-math",
        "-fno-signaling-nans",
        "-fcx-limited-range",
        "-fopenmp",
        "-fsimd-cost-model=unlimited",
    ]

    machine = platform.machine().lower()
    if machine == "x86_64":
        extra_compile_args += [
            "-mpopcnt",
            "-msse",
            "-msse2",
            "-msse3",
            "-mssse3",
            "-msse4.1",
            "-msse4.2",
            "-mavx",
            "-mavx2",
            "-mfma",
            "-mbmi",
            "-mbmi2",
            "-maes",
            "-mpclmul",
            "-mcx16",
            "-msahf",
            "-mf16c",
            "-mxsave",
            "-mfsgsbase",
            "-mlzcnt",
            "-mmovbe",
            "-mxsavec",
            "-mxsaves",
            "-mprfchw",
            "-mxsaveopt",
            "-mclflushopt",
            "-madx",
            "-mtune=generic",
        ]
    elif machine in ("aarch64", "armv8l"):
        extra_compile_args += ["-march=armv8-a+simd", "-mtune=cortex-a76"]
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")
elif sys.platform.startswith("win32"):
    raise RuntimeError("Windows is not supported for pip install; use packaging wheels or build.sh")
else:
    raise RuntimeError("Unsupported platform")


def get_pkg_config_flags(packages, flag):
    try:
        cmd = ["pkg-config", flag, "--static"] + packages
        return subprocess.check_output(cmd).decode("utf-8").strip().split()
    except subprocess.CalledProcessError:
        print(f"Error: pkg-config failed for {packages}. Is OpenCV installed?")
        sys.exit(1)


os.environ["PKG_CONFIG_PATH"] = (
    os.environ.get("PKG_CONFIG_PATH", "")
    + ":/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig"
)

opencv_includes = [
    f.replace("-I", "") for f in get_pkg_config_flags(["opencv4"], "--cflags-only-I")
]
opencv_link_args = get_pkg_config_flags(["opencv4"], "--libs")

setup(
    ext_modules=[
        Pybind11Extension(
            "mxprepost",
            [str(ROOT / "pymodule/bindings.cpp")]
            + sorted(str(p) for p in (ROOT / "src").glob("*.cpp")),
            include_dirs=[np.get_include(), str(ROOT / "include")] + opencv_includes,
            extra_compile_args=extra_compile_args,
            cxx_std=17,
            extra_link_args=["-lmemx", "-lmx_accl"] + opencv_link_args,
        )
    ],
    scripts=[str(ROOT / "samples/python/run.py")],
)
