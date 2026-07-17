# ===========================================================================
# CHPC-ready build for the Hedgehog LPBF solver + additive single-GPU CUDA port
#
# CPU-only (default, unchanged behaviour — builds exactly as before):
#     cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
#     cmake --build build -j
#     ./build/hedgehog_lpbf --case=0 --threads=4 --tiles=8
#
# With the CUDA port (needs a CUDA toolkit + nvcc):
#     cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
#           -DHEDGEHOG_LPBF_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
#     cmake --build build -j
#     ./build/hedgehog_lpbf_gpu --case=0 --check
#
# Hedgehog v3.3.3 is vendored under third_party/hedgehog — no external install
# is required; HEDGEHOG_INCLUDE_DIR defaults to it.
# ===========================================================================
cmake_minimum_required(VERSION 3.20)
project(hedgehog_lpbf CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Hedgehog (vendored, header-only) ---------------------------------------
include(FetchContent)
FetchContent_Declare(
    hedgehog
    GIT_REPOSITORY https://github.com/usnistgov/hedgehog.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(hedgehog)

FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        5.1.1
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)
include(${kokkos_BINARY_DIR}/KokkosConfigCommon.cmake) #FIXME: https://github.com/kokkos/kokkos/issues/7359#event-14339616469


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# std::filesystem lives in a separate libstdc++fs only on GCC < 9 (still a valid
# host compiler for CUDA 12.x).  On GCC >= 9 / Clang the generator expression
# evaluates to nothing, so this is a no-op on modern toolchains.
set(STDCXXFS_IF_NEEDED
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>)

# --- CPU executable (the primary single-track solver) -----------------------
add_executable(hedgehog_lpbf src/main.cpp)
target_include_directories(hedgehog_lpbf PRIVATE include)
target_link_libraries(hedgehog_lpbf PRIVATE hedgehog::hedgehog Threads::Threads
    ${STDCXXFS_IF_NEEDED})
target_compile_options(hedgehog_lpbf PRIVATE -Wall -Wextra -Wpedantic -O3)

# --- Additive CUDA port -----------------------------------------------------
# OFF by default: the CPU target above builds identically whether or not a CUDA
# toolkit is present.  Turning this ON adds the hedgehog_lpbf_gpu target only.
option(HEDGEHOG_LPBF_CUDA "Build the single-GPU CUDA port (needs nvcc)" OFF)

add_executable(kokkos_lpbf src/kokkos/lbpf.cpp)
target_link_libraries(kokkos_lpbf PUBLIC Kokkos::kokkos)
target_include_directories(kokkos_lpbf PRIVATE include)

if(HEDGEHOG_LPBF_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # A100=80, RTX3090/A6000=86, H100=90.  Add 89 (L40S/RTX6000-Ada) or 75
        # (T4) / 70 (V100) as needed for the CHPC node you target.
        set(CMAKE_CUDA_ARCHITECTURES "80;86;90")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "HEDGEHOG_LPBF_CUDA=ON but no CUDA compiler (nvcc) was found. "
            "Load a CUDA toolkit (on CHPC: `module load cuda`) or point CMake "
            "at nvcc with -DCMAKE_CUDA_COMPILER=$(which nvcc). "
            "The CPU targets do not require CUDA — configure without "
            "-DHEDGEHOG_LPBF_CUDA=ON to build them alone.")
    endif()
    enable_language(CUDA)
    # The .cu files deliberately do NOT include Hedgehog, so nvcc never sees the
    # Hedgehog C++20 (concepts/requires).  C++17 is sufficient for the kernels.
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDAToolkit gives us CUDA::cudart + the toolkit include dir.  The latter is
    # needed by the g++-compiled Hedgehog+CUDA TU: <hedgehog/hedgehog.h> under
    # HH_USE_CUDA transitively includes <cuda_runtime.h> and <cublas_v2.h>.
    find_package(CUDAToolkit REQUIRED)

    option(HEDGEHOG_LPBF_STRICT_FMAD "nvcc --fmad=false (tighter --check)" OFF)

    # --- Shared device object: the ONE TU with __global__ kernel bodies -------
    # Both the standalone GPU binary and the Hedgehog+CUDA binary link this, so
    # there is a single definition of step_kernel / reduce_kernel.
    add_library(lpbf_kernels OBJECT src/gpu/kernels.cu)
    target_include_directories(lpbf_kernels PRIVATE include)
    set_target_properties(lpbf_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    if(HEDGEHOG_LPBF_STRICT_FMAD)
        target_compile_options(lpbf_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()

    # --- Standalone nvcc reference/bench/check binary (refactored) ------------
    add_executable(hedgehog_lpbf_gpu src/gpu/lpbf_cuda.cu)
    target_include_directories(hedgehog_lpbf_gpu PRIVATE include)
    target_link_libraries(hedgehog_lpbf_gpu PRIVATE lpbf_kernels ${STDCXXFS_IF_NEEDED})
    set_target_properties(hedgehog_lpbf_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)
    if(HEDGEHOG_LPBF_STRICT_FMAD)
        target_compile_options(hedgehog_lpbf_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()

    # --- Hedgehog + CUDA dataflow binary (compiled by g++, C++20) -------------
    # A plain CXX target that #includes <cuda_runtime.h> and hh::AbstractCUDATask
    # (pure host code).  It does NOT enable_language(CUDA) for itself; it links
    # the kernels object above plus CUDA::cudart (cublas is header-only here and
    # never odr-used, so it is not linked).
    add_executable(hedgehog_lpbf_hhcuda src/gpu/hh_gpu_main.cpp)
    target_include_directories(hedgehog_lpbf_hhcuda PRIVATE
        include ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(hedgehog_lpbf_hhcuda PRIVATE
        HH_USE_CUDA HH_ENABLE_CHECK_CUDA)
    target_compile_options(hedgehog_lpbf_hhcuda PRIVATE -Wall -Wextra -O3)
    target_link_libraries(hedgehog_lpbf_hhcuda PRIVATE
        hedgehog::hedgehog Threads::Threads CUDA::cudart lpbf_kernels
        ${STDCXXFS_IF_NEEDED})

    message(STATUS "CUDA port ON  (arch=${CMAKE_CUDA_ARCHITECTURES})")
endif()
