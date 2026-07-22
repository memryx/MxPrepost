#!/bin/bash

# This script extracts the data from memx-accl_${ACCL_VER}_{amd64,arm64}.deb and memx-drivers_${DRIVER_VER}_{amd64,arm64}.deb, and puts them in respective pkg_ext_x86 and pkg_ext_arm folders.

DRIVER_VER=${1}
ACCL_VER=${2}

if [ -z ${1} ] || [ -z ${2} ]; then
  echo -e "Usage: ./extract_mxapi_pymod_deps.sh \033[95mDRIVER_VER\033[0m \033[96mACCL_VER\033[0m"
  exit 1
fi

# check that the files exist
WAS_DOWNLOADED=0
if [ ! -f memx-accl_${ACCL_VER}_amd64.deb ] || [ ! -f memx-accl_${ACCL_VER}_arm64.deb ] || [ ! -f memx-drivers_${DRIVER_VER}_amd64.deb ] || [ ! -f memx-drivers_${DRIVER_VER}_arm64.deb ]; then

  # try to download from developer.memryx.com/deb/pool/main/
  BASE_URL="https://developer.memryx.com/deb/pool/main/"
  wget "${BASE_URL}memx-accl_${ACCL_VER}_amd64.deb" &&
  wget "${BASE_URL}memx-accl_${ACCL_VER}_arm64.deb" &&
  wget "${BASE_URL}memx-drivers_${DRIVER_VER}_amd64.deb" &&
  wget "${BASE_URL}memx-drivers_${DRIVER_VER}_arm64.deb"
  if [ $? -ne 0 ]; then
    echo "Error: Required versions not found in stable devhub. Trying devtest..."
    BASE_URL="https://devtest.memryx.com/deb/pool_testing/main/"
    wget "${BASE_URL}memx-accl_${ACCL_VER}_amd64.deb" &&
    wget "${BASE_URL}memx-accl_${ACCL_VER}_arm64.deb" &&
    wget "${BASE_URL}memx-drivers_${DRIVER_VER}_amd64.deb" &&
    wget "${BASE_URL}memx-drivers_${DRIVER_VER}_arm64.deb"
    if [ $? -ne 0 ]; then
      echo "Error: Required versions not found in devtest either. Giving up. Download them to this folder and try again!"
      exit 1
    else
      echo "Gottem from testing! Continuing."
      WAS_DOWNLOADED=1
    fi
  else
    echo "Gottem from stable! Continuing."
    WAS_DOWNLOADED=1
  fi
fi

# delete old
rm -rf pkg_ext_x86 pkg_ext_arm

# create a temp directory to extract the deb files
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"
mkdir -p "$TEMP_DIR/amd64" "$TEMP_DIR/arm64"

# extract the deb files
dpkg-deb -x memx-accl_${ACCL_VER}_amd64.deb "$TEMP_DIR/amd64"
dpkg-deb -x memx-accl_${ACCL_VER}_arm64.deb "$TEMP_DIR/arm64"
dpkg-deb -x memx-drivers_${DRIVER_VER}_amd64.deb "$TEMP_DIR/amd64"
dpkg-deb -x memx-drivers_${DRIVER_VER}_arm64.deb "$TEMP_DIR/arm64"
echo "Extracted .deb files into temporary directory."

# create the output directories
OUTPUT_DIR_X86="pkg_ext_x86"
OUTPUT_DIR_ARM="pkg_ext_arm"
mkdir -p "$OUTPUT_DIR_X86" "$OUTPUT_DIR_ARM"
echo "Created output directories: $OUTPUT_DIR_X86 and $OUTPUT_DIR_ARM"

# copy the necessary files to the output directories
# 1. we want /usr/lib/$ARCH-linux-gnu/libmemx.so* and /usr/lib/$ARCH-linux-gnu/libmx_accl.so* in the top pkg_ext_$ARCH directory
# 2. we want /usr/include/memx/* in the pkg_ext_$ARCH/memx/ directory

cp -v "$TEMP_DIR/amd64/usr/lib/x86_64-linux-gnu/libmemx.so"* "$OUTPUT_DIR_X86/"
cp -v "$TEMP_DIR/amd64/usr/lib/x86_64-linux-gnu/libmx_accl.so"* "$OUTPUT_DIR_X86/"
mkdir -p "$OUTPUT_DIR_X86/memx"
cp -vr "$TEMP_DIR/amd64/usr/include/memx/"* "$OUTPUT_DIR_X86/memx/"
echo "Copied files to $OUTPUT_DIR_X86"

cp -v "$TEMP_DIR/arm64/usr/lib/aarch64-linux-gnu/libmemx.so"* "$OUTPUT_DIR_ARM/"
cp -v "$TEMP_DIR/arm64/usr/lib/aarch64-linux-gnu/libmx_accl.so"* "$OUTPUT_DIR_ARM/"
mkdir -p "$OUTPUT_DIR_ARM/memx"
cp -vr "$TEMP_DIR/arm64/usr/include/memx/"* "$OUTPUT_DIR_ARM/memx/"
echo "Copied files to $OUTPUT_DIR_ARM"

rm -rf "$TEMP_DIR"
if [[ $WAS_DOWNLOADED -eq 1 ]]; then
  rm -f memx-accl_${ACCL_VER}_amd64.deb memx-accl_${ACCL_VER}_arm64.deb memx-drivers_${DRIVER_VER}_amd64.deb memx-drivers_${DRIVER_VER}_arm64.deb
fi

echo -e "\033[92mDONE\033[0m"
exit 0
