import os
import sys
import fnmatch
import sysconfig
import numpy as np
import hashlib
from glob import glob
from pathlib import Path
import subprocess

from setuptools import setup, find_packages, Extension

from pybind11.setup_helpers import Pybind11Extension, build_ext

import platform


PKG_VERSION=Path('VERSION').read_text()

if sys.platform.startswith('linux'):
    # tons of optimizations, copied from CMakeLists.txt
    extra_compile_args=['-O3','-std=c++17','-fno-math-errno','-funsafe-math-optimizations',
                        '-ftree-vectorize','-ftree-loop-ivcanon', '-ftree-loop-im',
                        '-ffinite-math-only','-fno-signed-zeros',
                        '-fno-trapping-math','-fno-signaling-nans',
                        '-fcx-limited-range','-fopenmp',
                        '-fsimd-cost-model=unlimited']

    if str(platform.machine()).lower() == 'x86_64':
        # covers Haswell / Zen 1 and later
        # if you're using that Intel N5105 thing... just build from source
        extra_compile_args += ['-mpopcnt','-msse','-msse2','-msse3','-mssse3','-msse4.1','-msse4.2','-mavx','-mavx2','-mfma','-mbmi','-mbmi2','-maes','-mpclmul','-mcx16','-msahf','-mf16c','-mxsave','-mfsgsbase','-mlzcnt','-mmovbe','-mxsavec','-mxsaves','-mprfchw','-mxsaveopt','-mclflushopt','-madx','-mtune=generic']

    elif str(platform.machine()).lower() == 'aarch64' or str(platform.machine()).lower() == 'armv8l':
        # raspberry pi 5 or RK3588 are pretty much every common board we find, so optimize for A76 SIMD
        extra_compile_args += ['-march=armv8-a+simd','-mtune=cortex-a76']

elif sys.platform.startswith('win32'):
    extra_link_args=['udriver.lib',
                   '/LIBPATH:..\\udriver\\build']
    extra_compile_args=['/O2','/arch:AVX2']
else:
    extra_link_args=[]
    extra_compile_args=[]
    raise RuntimeError("Unsupported platform")


def get_pkg_config_flags(packages, flag):
    """Retrieve flags from pkg-config."""
    try:
        # static libs of any deps
        cmd = ["pkg-config", flag, "--static"] + packages
        return subprocess.check_output(cmd).decode("utf-8").strip().split()
    except subprocess.CalledProcessError:
        print(f"Error: pkg-config failed for {packages}. Is OpenCV installed?")
        sys.exit(1)

# update PKG_CONFIG_PATH to find the OpenCV install
os.environ["PKG_CONFIG_PATH"] = (
    os.environ.get("PKG_CONFIG_PATH", "") + ":/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig"
)

opencv_includes = [f.replace("-I", "") for f in get_pkg_config_flags(["opencv4"], "--cflags-only-I")]
opencv_link_args = get_pkg_config_flags(["opencv4"], "--libs")


setup(
    name='mxprepost',
    version=PKG_VERSION,
    url="https://developer.memryx.com",
    author="MemryX",
    author_email="no-reply@memryx.com",
    description="Optimized Pre/Post for models on MemryX MX3",
    ext_modules=[Pybind11Extension("mxprepost",
                           ["pymodule/bindings.cpp"] + sorted(glob("src/*.cpp")),
                           include_dirs=[np.get_include(), "include"] + opencv_includes,
                           extra_compile_args=extra_compile_args,
                           cxx_std=17,
                           extra_link_args=['-lmemx','-lmx_accl'] + opencv_link_args 
                           )],
    license="MIT",
    license_files=["LICENSE"],
    scripts=['mx_yolo_test'],
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[
        'pybind11~=2.13.6',
        'opencv-python~=4.11.0',
        'numpy~=1.26.4'
    ]
)
