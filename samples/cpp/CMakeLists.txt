cmake_minimum_required(VERSION 3.10)
project(yolo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(CMAKE_X86_FLAGS "-mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mfxsr -mcx16 -msahf -mpclmul -mfsgsbase -mmovbe -mprfchw -mxsave -mxsaveopt -mavx -mavx2 -mfma -mbmi -mbmi2 -maes -mf16c -mlzcnt -madx -mxsavec -mxsaves -mclflushopt -mtune=generic")
set(CMAKE_ARM_FLAGS "-march=armv8-a+simd -mtune=cortex-a76")

# Compiler flags
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O0 -g -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -fopenmp -DNDEBUG -ftree-vectorize -fsimd-cost-model=unlimited -fno-math-errno -funsafe-math-optimizations -ftree-loop-ivcanon -ftree-loop-im -ffinite-math-only -fno-signed-zeros -fno-trapping-math -fno-signaling-nans -fcx-limited-range")

    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_X86_FLAGS}")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ARM_FLAGS}")
    else()
        message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}. No specific optimization flags will be set.")
    endif()
    
endif()

file(GLOB local_src "*.cpp" "*.c")

# Get first cpp file and extract filename without extension
list(GET local_src 0 MAIN_SRC)
get_filename_component(APP_NAME ${MAIN_SRC} NAME_WE)

add_executable(${APP_NAME} ${local_src})

target_include_directories(${APP_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ROOT_DIR}/include
)

target_link_directories(${APP_NAME} PRIVATE
    ${ROOT_DIR}/build
)

target_link_libraries(${APP_NAME} PRIVATE
    mx_accl
    mxprepost
    ${OpenCV_LIBS}
    Threads::Threads
    dl
)

set_target_properties(${APP_NAME} PROPERTIES
    BUILD_RPATH "${ROOT_DIR}/build"
)
