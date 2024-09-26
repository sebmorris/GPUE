cmake_minimum_required(VERSION 3.8)
project(gpue_build LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_file( EXAMPLE_CFG example.cfg src/ )
add_library(unitTest OBJECT src/unit_test.cu include/unit_test.h include/kernels.h)
set( CFG_FILE_PATH _CFG_FILE_PATH="${EXAMPLE_CFG}" )
target_compile_options(
   unitTest PRIVATE -D${CFG_FILE_PATH} -dc -std=c++11
)

add_executable(
    gpue
    src/dynamic.cu
    include/dynamic.h

    src/ds.cu
    include/ds.h

    src/edge.cu
    include/edge.h

    src/evolution.cu
    include/evolution.h

    src/fileIO.cu
    include/fileIO.h

    src/kernels.cu
    include/kernels.h

    src/manip.cu
    include/manip.h

    src/lattice.cu
    include/lattice.h

    src/minions.cu
    include/minions.h

    src/multigpu.cu

    src/node.cu
    include/node.h

    src/operators.cu
    include/operators.h

    src/parser.cu
    include/parser.h

    src/split_op.cu
    include/split_op.h

    src/tracker.cu
    include/tracker.h

    $<TARGET_OBJECTS:unitTest>

    src/vort.cu
    include/vort.h

    src/vortex_3d.cu
    include/vortex_3d.h

    src/init.cu
    include/init.h

    src/XGetopt.cpp
    include/XGetopt.h
)

target_compile_features(gpue PUBLIC cxx_std_11)
find_package(CUDAToolkit)
target_link_libraries(gpue CUDA::cufft)
set_target_properties(gpue PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
