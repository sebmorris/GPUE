cmake_minimum_required(VERSION 2.8)
project(gpue_build LANGUAGES CXX CUDA)
find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_37,code=sm_37 -lcufft -fopenmp
)
#add_library(
add_executable(
    gpue
    src/ds.cc
    include/ds.h

    src/dynamic.cu
    include/dynamic.h

    src/edge.cc
    include/edge.h

    src/evolution.cu
    include/evolution.h

    src/fileIO.cc
    include/fileIO.h

    src/kernels.cu
    include/kernels.h

    src/manip.cu
    include/manip.h

    src/lattice.cc
    include/lattice.h

    src/minions.cc
    include/minions.h

    src/multigpu.cu

    src/node.cc
    include/node.h

    src/operators.cu
    include/operators.h

    src/parser.cc
    include/parser.h

    src/split_op.cu
    include/split_op.h

    src/tracker.cc
    include/tracker.h

    src/unit_test.cu
    include/unit_test.h

    src/vort.cc
    include/vort.h

    src/vortex_3d.cu
    include/vortex_3d.h

    src/init.cu
    include/init.h
)

target_compile_features(gpue PUBLIC cxx_std_11)
target_link_libraries(gpue ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
set_target_properties(gpue PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
