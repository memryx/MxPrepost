cmake_minimum_required(VERSION 3.14)
project(mxprepost)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

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


file(GLOB local_src src/*.cpp)

find_package(OpenCV REQUIRED)


add_library(mxprepost SHARED ${local_src})

target_include_directories(mxprepost PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    include/
)

target_link_libraries(mxprepost ${OpenCV_LIBS} mx_accl)


add_subdirectory(samples/cpp)
