# RexGraph CUDA kernel build.
#
# This file is not the default build system; the default is
# meson-python (see meson.build). Use this only to compile
# GPU-accelerated kernels:
#
#   cd rexgraph/cuda && mkdir build && cd build
#   cmake .. -DCUDA_ARCH="80;86;89;90"
#   make -j$(nproc)
#   cp *.so ../
#
# The resulting shared library is loaded at runtime by
# rexgraph.cuda. If absent, rexgraph falls back to CPU.
#
# Compiled operations:
#   - CSR sparse matrix-vector multiply (Laplacians)
#   - Batched eigendecomposition (cuSOLVER)
#   - PageRank power iteration
#   - Force-directed layout

cmake_minimum_required(VERSION 3.24)
project(rexgraph_cuda LANGUAGES CXX CUDA)

# Target architectures. Default: Ampere (80), Ada Lovelace (89), Hopper (90).
# Override with -DCUDA_ARCH="75;80;86;89;90".
set(CUDA_ARCH "80;89;90" CACHE STRING "CUDA compute capabilities to target")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Python (for NumPy array interop via pybind11 or ctypes)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# All kernels are compiled into a single shared library.
# Python loads it via ctypes or a thin Cython wrapper.

add_library(rexgraph_cuda_kernels SHARED
    src/spmv_csr.cu          # sparse matrix-vector multiply
    src/eigensolver.cu       # eigenvalue computation
    src/pagerank_gpu.cu      # PageRank power iteration
    src/force_layout_gpu.cu  # force-directed layout
)

target_link_libraries(rexgraph_cuda_kernels PRIVATE
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
    CUDA::curand
)

target_include_directories(rexgraph_cuda_kernels PRIVATE
    ${Python_INCLUDE_DIRS}
    include/
)

# Install
install(
    TARGETS rexgraph_cuda_kernels
    LIBRARY DESTINATION rexgraph/cuda
    RUNTIME DESTINATION rexgraph/cuda
)
