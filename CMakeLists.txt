cmake_minimum_required(VERSION 3.10)
project(Reverse-kRanks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green "${Esc}[32m")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("${Green}Release mode${ColourReset}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("${Green}Clang${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("${Green}GNU${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("${Green}MSVC${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS})
    else ()
        message("${Green}Unknown compiler${ColourReset}")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 ${OpenMP_CXX_FLAGS}")
    message("${Green}Debug mode${ColourReset}")
else ()
    message("${Green}Unknown mode${ColourReset}")
endif ()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")


find_package(Boost COMPONENTS program_options)

#include(FetchContent)
#FetchContent_Declare(
#        spdlog
#        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.10.0.tar.gz
#)
#FetchContent_MakeAvailable(spdlog)

find_package(spdlog REQUIRED)

set(BUILD_STATIC_LIBS ON)
set(BUILD_WITHOUT_LAPACK OFF)

find_package(BLAS REQUIRED)

set(OpenBLAS_LIBRARY ${openblas_BINARY_DIR}/lib/libopenblas.a)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(DETECT_HDF5 OFF CACHE BOOL "")
set(OPENBLAS_PROVIDES_LAPACK ON CACHE BOOL "")
set(ALLOW_FLEXIBLAS_LINUX OFF CACHE BOOL "")
set(ALLOW_OPENBLAS_MACOS ON CACHE BOOL "")

find_package(Armadillo)

find_package(OpenMP REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(fexipro
        src/include/fexipro/alg/Naive.h
        src/include/fexipro/alg/SVDIntUpperBoundIncrPrune2.h

        src/include/fexipro/structs/ExtendMatrix.h
        src/include/fexipro/structs/ExtendMatrixRow.h
        src/include/fexipro/structs/FastHeap.h
        src/include/fexipro/structs/IntMatrixRow.h
        src/include/fexipro/structs/Matrix.h
        src/include/fexipro/structs/SIRMatrixRow.h
        src/include/fexipro/structs/SVDIntMatrixRow.h
        src/include/fexipro/structs/VectorElement.h

        src/include/fexipro/util/Base.h
        src/include/fexipro/util/Calculator.h
        src/include/fexipro/util/Conf.h
        src/include/fexipro/util/FileUtil.h
        src/include/fexipro/util/Logger.h
        src/include/fexipro/util/Monitor.h
        src/include/fexipro/util/SVDUtil.h
        src/include/fexipro/util/TransformUtil.h)

add_library(fexiprolib SHARED ${fexipro})
set_target_properties(fexiprolib PROPERTIES LINKER_LANGUAGE CXX)
#target_link_directories(fexiprolib PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES})
target_link_libraries(fexiprolib PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES})
target_include_directories(fexiprolib PRIVATE ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

set(rtk
        src/include/rtree/dyy_data.cpp
        src/include/rtree/dyy_RstarTree.cpp
        src/include/rtree/dyy_RTK_method.cpp)

add_library(rtklib SHARED ${rtk})
target_link_libraries(rtklib PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(rtklib PROPERTIES LINKER_LANGUAGE CXX)

set(USE_CUDA OFF)

if (USE_CUDA)

    set(USE_RETRIEVAL_GPU ON)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_definitions(-DUSE_GPU)

    if (USE_RETRIEVAL_GPU)
        add_definitions(-DUSE_RETRIEVAL_GPU)
    endif ()

    #    find_package(CUDAToolkit)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    enable_language(CUDA)
    find_package(CUDA)

    #    set(Thrust_DIR /usr/include/thrust)
    find_package(Thrust REQUIRED CONFIG)
    thrust_create_target(Thrust)

    set_source_files_properties(src/app/BuildScoreTableBatch.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(bstb src/app/BuildScoreTableBatch.cpp)
    target_include_directories(bstb PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(bstb PRIVATE armadillo BLAS::BLAS pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES} cublas)
    set_target_properties(bstb PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
    set_property(TARGET bstb PROPERTY CXX_STANDARD "17")
    set_property(TARGET bstb PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET bstb PROPERTY CXX_EXTENSIONS OFF)
    set_target_properties(bstb PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    target_compile_options(bstb PRIVATE -Xcompiler=-fopenmp)

    set_source_files_properties(src/app/build-index/QueryDistributionIndexByComputation.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(qdibc src/app/build-index/QueryDistributionIndexByComputation.cpp)
    target_include_directories(qdibc PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(qdibc PRIVATE armadillo BLAS::BLAS pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES} cublas)
    set_target_properties(qdibc PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
    set_property(TARGET qdibc PROPERTY CXX_STANDARD "17")
    set_property(TARGET qdibc PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET qdibc PROPERTY CXX_EXTENSIONS OFF)
    set_target_properties(qdibc PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    target_compile_options(qdibc PRIVATE -Xcompiler=-fopenmp)

    set_source_files_properties(src/app/build-index/BuildSampleIndexByComputation.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(bsibc src/app/build-index/BuildSampleIndexByComputation.cpp)
    target_include_directories(bsibc PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(bsibc PRIVATE armadillo BLAS::BLAS pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES} cublas)
    set_target_properties(bsibc PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
    set_property(TARGET bsibc PROPERTY CXX_STANDARD "17")
    set_property(TARGET bsibc PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET bsibc PROPERTY CXX_EXTENSIONS OFF)
    set_target_properties(bsibc PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    target_compile_options(bsibc PRIVATE -Xcompiler=-fopenmp)

    set_source_files_properties(src/app/build-index/BuildRegressionIndex.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(bri src/app/build-index/BuildRegressionIndex.cpp)
    target_include_directories(bri PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(bri PRIVATE armadillo BLAS::BLAS pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES} cublas Eigen3::Eigen)
    set_target_properties(bri PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
    set_property(TARGET bri PROPERTY CXX_STANDARD "17")
    set_property(TARGET bri PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET bri PROPERTY CXX_EXTENSIONS OFF)
    set_target_properties(bri PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    target_compile_options(bri PRIVATE -Xcompiler=-fopenmp)

else () # not use cuda

    add_executable(bstb src/app/BuildScoreTableBatch.cpp)
    target_include_directories(bstb PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
    target_link_libraries(bstb PRIVATE armadillo pthread spdlog::spdlog OpenMP::OpenMP_CXX ${Boost_LIBRARIES})

    add_executable(qdibc src/app/build-index/QueryDistributionIndexByComputation.cpp)
    target_include_directories(qdibc PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
    target_link_libraries(qdibc PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES})

    add_executable(bsibc src/app/build-index/BuildSampleIndexByComputation.cpp)
    target_include_directories(bsibc PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
    target_link_libraries(bsibc PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES})

    add_executable(bri src/app/build-index/BuildRegressionIndex.cpp)
    target_include_directories(bri PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
    target_link_libraries(bri PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog Eigen3::Eigen ${Boost_LIBRARIES})

endif ()

add_executable(qdibs src/app/build-index/QueryDistributionIndexBySample.cpp)
target_include_directories(qdibs PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
target_link_libraries(qdibs PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES})

add_executable(bsvdi src/app/build-index/BuildSVDIndex.cpp)
target_include_directories(bsvdi PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
target_link_libraries(bsvdi PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog Eigen3::Eigen ${Boost_LIBRARIES} fexiprolib rtklib)

add_executable(rri src/app/ReverseRankIndex.cpp)
target_include_directories(rri PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} src/include/LEMP)
target_link_libraries(rri PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog Eigen3::Eigen ${Boost_LIBRARIES} fexiprolib rtklib)

add_executable(rriu src/app/ReverseRankIndexUpdate.cpp)
target_include_directories(rriu PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} src/include/LEMP)
target_link_libraries(rriu PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog Eigen3::Eigen ${Boost_LIBRARIES} fexiprolib rtklib)

add_executable(fsr src/app/build-index/FindSampleRank.cpp)
target_include_directories(fsr PRIVATE src/include src/impl ${armadillo_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})
target_link_libraries(fsr PRIVATE armadillo pthread OpenMP::OpenMP_CXX spdlog::spdlog ${Boost_LIBRARIES})

