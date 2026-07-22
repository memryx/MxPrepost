#!/bin/bash

if ! [[ -f setup.py ]]; then
  echo "setup.py not found. Are you in the right folder?"
  exit 1
fi


# detect if this host is x86 or ARM and normalize to "x86" and "arm"
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]] || [[ "$ARCH" == "amd64" ]]; then
  ARCH="x86"
elif [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
  ARCH="arm"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi


# make sure pkg_ext_$ARCH exists
if ! [[ -d "pkg_ext_$ARCH" ]]; then
  echo "Directory pkg_ext_$ARCH not found. Please run extract_deps.sh first to create it."
  exit 1
fi


#
# STEP 1: COPY FILES TO DOCKER SHARE
#

echo "Copying..."

# external deps (pybind11)
cp -Lr ../extern ./

# mxprepost c++ code
cp -Lr ../include ./
cp -Lr ../src ./

# pymodule source but leave pybind11 as a symlink
cp -a ../pymodule ./

cp -v ../samples/python/run.py ./mx_yolo_test
chmod +x mx_yolo_test

#
# STEP 2: BUILD THE PIP
#

# used later
MY_UID=$(id -u)

# start the relevant docker (by platform) and call run_me_in_docker.sh in it
if [[ $ARCH == "x86" ]]; then
  docker run --platform=linux/amd64 -it -v $(pwd):/io quay.io/pypa/manylinux_2_28_x86_64 /bin/bash -c "/io/run_me_in_docker.sh ${MY_UID}"
elif [[ $ARCH == "arm" ]]; then
  docker run --platform=linux/arm64/v8 -it -v $(pwd):/io quay.io/pypa/manylinux_2_28_aarch64 /bin/bash -c "/io/run_me_in_docker.sh ${MY_UID}"
fi


if [ $? -ne 0 ]; then
  echo "Build failed!"
  if [[ ${1} == "nosave" ]]; then
    rm -rf dist
    rm -rf mxprepost*
    rm -rf build
    rm -rf extern
    rm -rf include
    rm -rf pymodule
    rm -rf src
    rm -rf wheelhouse
    rm -f mx_yolo_test
  fi
  exit 1
fi

if [[ ${1} == "nosave" ]]; then
  rm -rf dist
  rm -rf mxprepost*
  rm -rf build
  rm -rf extern
  rm -rf include
  rm -rf pymodule
  rm -rf src
  rm -f mx_yolo_test
  mv wheelhouse output
  exit 0
else

  read -p "Save build files? [y/N] "
  
  if [[ $REPLY == "y" ]] || [[ $REPLY == "Y" ]] || [[ $REPLY == "yes" ]]; then
    mv wheelhouse output
    exit 0
  else
    rm -rf dist
    rm -rf mxprepost*
    rm -rf build
    rm -rf extern
    rm -rf include
    rm -rf pymodule
    rm -rf src
    rm -f mx_yolo_test
    mv wheelhouse output
    exit 0
  fi

fi
