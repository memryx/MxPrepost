#!/bin/bash

MY_UID=${1}

function clean_exit {
  chown -hR ${MY_UID} /io
  exit 0
}

function dirty_exit {
  chown -hR ${MY_UID} /io
  exit 1
}



if ! [[ -d /io ]]; then
  echo "ERROR: /io not mounted"
  exit 1
fi

cd /io
PKG_VERSION=$(cat VERSION)


###############################################################################################

# detect if this host is x86 or ARM and normalize to "x86" and "arm"
ARCH=$(uname -m)
ARCHLONG=$ARCH
if [[ "$ARCH" == "x86_64" ]] || [[ "$ARCH" == "amd64" ]]; then
  ARCH="x86"
  ARCHLONG="x86_64"
elif [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
  ARCH="arm"
  ARCHLONG="aarch64"
else
  echo "run_me_in_docker: unsupported architecture: $ARCH"
  exit 1
fi


# make sure pkg_ext_$ARCH exists
if ! [[ -d "pkg_ext_$ARCH" ]]; then
  echo "run_me_in_docker: directory pkg_ext_$ARCH not found. Please run extract_deps.sh first to create it."
  exit 1
fi

###############################################################################################

# build opencv from source...
dnf install -y cmake gcc-c++ gcc-toolset-15-gcc gcc-toolset-15-gcc-c++ git wget \
    libjpeg-devel libpng-devel libtiff-devel \
    zlib-devel bzip2-devel

# 2. Setup Directories
OPENCV_VERSION="4.11.0"
WORKING_DIR="/opt/opencv_build"
INSTALL_DIR="/usr/local"

#export CC="/opt/rh/gcc-toolset-15/root/bin/gcc"
#export CXX="/opt/rh/gcc-toolset-15/root/bin/g++"

mkdir -p $WORKING_DIR
cd $WORKING_DIR

# 3. Clone OpenCV
git clone --depth 1 --branch $OPENCV_VERSION https://github.com/opencv/opencv.git

cd opencv

# --- COMPATIBILITY PATCH FOR CMAKE 3.27+ ---
echo "Patching all CMake/Script requirements below 3.5..."
find . \( -name "CMakeLists.txt" -o -name "*.cmake" \) -exec sed -i -E \
    's/cmake_minimum_required\(VERSION (2\.[0-9.]+|3\.[0-4](\.[0-9.]+)?)\)/cmake_minimum_required(VERSION 3.5)/g' {} +

# 4. Create Build Directory
# -------------------------------------------

mkdir -p build
cd build

# 5. Configure CMake
TCFLAGS="-O3 -fopenmp -ftree-vectorize -fsimd-cost-model=unlimited -ftree-loop-ivcanon -ftree-loop-im"
BASELINE_DEF=""
if [[ $ARCH == "x86" ]]; then
  TCFLAGS="${TCFLAGS} -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mfxsr -mcx16 -msahf -mpclmul -mfsgsbase -mmovbe -mprfchw -mxsave -mxsaveopt -mavx -mavx2 -mfma -mbmi -mbmi2 -maes -mf16c -mlzcnt -madx -mxsavec -mxsaves -mclflushopt -mtune=generic"
  BASELINE_DEF="-DCPU_BASELINE=AVX2"
elif [[ $ARCH == "arm" ]]; then
  TCFLAGS="${TCFLAGS} -march=armv8-a+simd -mtune=cortex-a76"
fi

export CFLAGS=$TCFLAGS
export CXXFLAGS=$TCFLAGS

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D BUILD_SHARED_LIBS=OFF \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_LIST=core,dnn,imgproc \
      -D WITH_V4L=OFF \
      ${BASELINE_DEF} \
      -D WITH_FFMPEG=OFF \
      -D WITH_GSTREAMER=OFF \
      -D WITH_MSMF=OFF \
      -D WITH_WEBP=OFF \
      -D WITH_JPEG=OFF \
      -D WITH_PNG=OFF \
      -D WITH_TIFF=OFF \
      -D WITH_JASPER=OFF \
      -D WITH_OPENJPEG=OFF \
      -D WITH_OPENEXR=OFF \
      -D WITH_IMGCODEC_PXM=OFF \
      -D WITH_IMGCODEC_PFM=OFF \
      -D WITH_IMGCODEC_HDR=OFF \
      -D WITH_IMGCODEC_SUNRASTER=OFF \
      -D WITH_OPENCL=OFF \
      -D WITH_VA_INTEL=OFF \
      -D WITH_OPENGL=OFF \
      -D WITH_GTK=OFF \
      -D WITH_CUDA=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_java=OFF \
      -D WITH_IPP=OFF \
      ..

find .. \( -name "CMakeLists.txt" -o -name "*.cmake" \) -exec sed -i -E \
    's/cmake_minimum_required\(VERSION (2\.[0-9.]+|3\.[0-4](\.[0-9.]+)?)\)/cmake_minimum_required(VERSION 3.5)/g' {} +
# -------------------------------------------

# 6. Build and Install
if [[ $ARCH == "x86" ]]; then
  make -j16
else
  make -j6
fi
make install

cd /io

###############################################################################################

cp -v pkg_ext_$ARCH/libm* /usr/lib/ &&
cp -vr pkg_ext_$ARCH/memx/ /usr/include/memx/
if [ $? -ne 0 ]; then
  dirty_exit
fi

rm -rf build
/opt/python/cp39-cp39/bin/python -m pip install setuptools
/opt/python/cp39-cp39/bin/python -m pip install wheel
/opt/python/cp39-cp39/bin/python -m pip install "auditwheel==5.4.0"
/opt/python/cp39-cp39/bin/python -m pip install "numpy~=1.26.0"
/opt/python/cp39-cp39/bin/python -m pip install "pybind11~=2.13.6"
/opt/python/cp39-cp39/bin/python setup.py build_ext -j 4 &&
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel
if [ $? -eq 0 ]; then
  rm -rfv /usr/include/memx /usr/lib/libmemx* /usr/lib/libmx_accl*
  /opt/python/cp39-cp39/bin/python patchlibs.py repair dist/mxprepost-${PKG_VERSION}-cp39-cp39-linux_${ARCHLONG}.whl
  if [ $? -ne 0 ]; then
    dirty_exit
  fi
else
  dirty_exit
fi

###############################################################################################

cp -v pkg_ext_$ARCH/libm* /usr/lib/ &&
cp -vr pkg_ext_$ARCH/memx/ /usr/include/memx/
if [ $? -ne 0 ]; then
  dirty_exit
fi

rm -rf build
/opt/python/cp310-cp310/bin/python -m pip install setuptools
/opt/python/cp310-cp310/bin/python -m pip install wheel
/opt/python/cp310-cp310/bin/python -m pip install "auditwheel==5.4.0"
/opt/python/cp310-cp310/bin/python -m pip install "numpy~=1.26.0"
/opt/python/cp310-cp310/bin/python -m pip install "pybind11~=2.13.6"
/opt/python/cp310-cp310/bin/python setup.py build_ext -j 4 &&
/opt/python/cp310-cp310/bin/python setup.py bdist_wheel
if [ $? -eq 0 ]; then
  rm -rfv /usr/include/memx /usr/lib/libmemx* /usr/lib/libmx_accl*
  /opt/python/cp310-cp310/bin/python patchlibs.py repair dist/mxprepost-${PKG_VERSION}-cp310-cp310-linux_${ARCHLONG}.whl
  if [ $? -ne 0 ]; then
    dirty_exit
  fi
else
  dirty_exit
fi
 
###############################################################################################

cp -v pkg_ext_$ARCH/libm* /usr/lib/ &&
cp -vr pkg_ext_$ARCH/memx/ /usr/include/memx/
if [ $? -ne 0 ]; then
  dirty_exit
fi

rm -rf build
/opt/python/cp311-cp311/bin/python -m pip install setuptools
/opt/python/cp311-cp311/bin/python -m pip install wheel
/opt/python/cp311-cp311/bin/python -m pip install "auditwheel==5.4.0"
/opt/python/cp311-cp311/bin/python -m pip install "numpy~=1.26.0"
/opt/python/cp311-cp311/bin/python -m pip install "pybind11~=2.13.6"
/opt/python/cp311-cp311/bin/python setup.py build_ext -j 4 &&
/opt/python/cp311-cp311/bin/python setup.py bdist_wheel
if [ $? -eq 0 ]; then
  rm -rfv /usr/include/memx /usr/lib/libmemx* /usr/lib/libmx_accl*
  /opt/python/cp311-cp311/bin/python patchlibs.py repair dist/mxprepost-${PKG_VERSION}-cp311-cp311-linux_${ARCHLONG}.whl
  if [ $? -ne 0 ]; then
    dirty_exit
  fi
else
  dirty_exit
fi
 
###############################################################################################

cp -v pkg_ext_$ARCH/libm* /usr/lib/ &&
cp -vr pkg_ext_$ARCH/memx/ /usr/include/memx/
if [ $? -ne 0 ]; then
  dirty_exit
fi

rm -rf build
/opt/python/cp312-cp312/bin/python -m pip install setuptools
/opt/python/cp312-cp312/bin/python -m pip install wheel
/opt/python/cp312-cp312/bin/python -m pip install "auditwheel==5.4.0"
/opt/python/cp312-cp312/bin/python -m pip install "numpy~=1.26.0"
/opt/python/cp312-cp312/bin/python -m pip install "pybind11~=2.13.6"
/opt/python/cp312-cp312/bin/python setup.py build_ext -j 4 &&
/opt/python/cp312-cp312/bin/python setup.py bdist_wheel
if [ $? -eq 0 ]; then
  rm -rfv /usr/include/memx /usr/lib/libmemx* /usr/lib/libmx_accl*
  /opt/python/cp312-cp312/bin/python patchlibs.py repair dist/mxprepost-${PKG_VERSION}-cp312-cp312-linux_${ARCHLONG}.whl
  if [ $? -ne 0 ]; then
    dirty_exit
  fi
else
  dirty_exit
fi
 
###############################################################################################
 
clean_exit
