cmake_minimum_required(VERSION 3.14)
project(fusedann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_AVX2 "Enable AVX2/FMA optimizations" ON)
option(ENABLE_EFANNA "Build the EFANNA backend (dense only)" ON)
option(ENABLE_PARLAYANN "Build the ParlayANN backend (dense + sparse/filtering)" OFF)

# OpenMP is required by both backends.
find_package(OpenMP REQUIRED)

if (NOT ENABLE_EFANNA AND NOT ENABLE_PARLAYANN)
  message(FATAL_ERROR "Nothing to build: enable at least one of ENABLE_EFANNA or ENABLE_PARLAYANN.")
endif()

# -----------------------------------------------------------------------------
# EFANNA backend (dense only)
# -----------------------------------------------------------------------------
if (ENABLE_EFANNA)
  set(EFANNA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/efanna" CACHE PATH "Root directory of the EFANNA library")
  set(EFANNA_INCLUDE_DIR
      "${EFANNA_ROOT}/include"
      "${EFANNA_ROOT}")

  find_path(EFANNA_HEADER_PATH
    NAMES efanna2e/index_graph.h
    HINTS ${EFANNA_INCLUDE_DIR})

  if (NOT EFANNA_HEADER_PATH)
    message(FATAL_ERROR "Could not find efanna2e/index_graph.h under ${EFANNA_INCLUDE_DIR}. "
                        "Set EFANNA_ROOT to your EFANNA checkout, or pass -DENABLE_EFANNA=OFF to skip this backend.")
  endif()

  add_executable(fusedann_efanna fusedann_efanna.cpp)

  target_include_directories(fusedann_efanna PRIVATE ${EFANNA_INCLUDE_DIR})

  file(GLOB EFANNA_SOURCES
    "${EFANNA_ROOT}/src/index.cpp"
    "${EFANNA_ROOT}/src/index_graph.cpp"
    "${EFANNA_ROOT}/src/index_random.cpp")

  target_sources(fusedann_efanna PRIVATE ${EFANNA_SOURCES})

  target_compile_definitions(fusedann_efanna PRIVATE
    _mm128_loadu_ps=_mm_loadu_ps
    _mm128_mul_ps=_mm_mul_ps
    _mm128_add_ps=_mm_add_ps)

  target_link_libraries(fusedann_efanna
    PRIVATE
      OpenMP::OpenMP_CXX)

  target_compile_definitions(fusedann_efanna PRIVATE -DUSE_SSE)

  if (ENABLE_AVX2)
    target_compile_options(fusedann_efanna PRIVATE -mavx2 -mfma)
  endif()
endif()

# -----------------------------------------------------------------------------
# ParlayANN backend (dense + sparse/filtering)
# -----------------------------------------------------------------------------
set(PARLAYANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/parlayann" CACHE PATH "Root directory of the ParlayANN checkout")
set(PARLAYLIB_INCLUDE_DIR "${PARLAYANN_ROOT}/parlaylib" CACHE PATH "Directory that provides parlay headers")

if (ENABLE_PARLAYANN)
  find_package(Threads REQUIRED)

  find_path(PARLAYANN_HEADER_PATH
    NAMES algorithms/vamana/index.h
    HINTS ${PARLAYANN_ROOT})

  if (NOT PARLAYANN_HEADER_PATH)
    message(FATAL_ERROR "Could not find ParlayANN headers under ${PARLAYANN_ROOT}. Set PARLAYANN_ROOT to your ParlayANN checkout.")
  endif()

  find_path(PARLAYLIB_HEADER_PATH
    NAMES parlay/parallel.h
    HINTS ${PARLAYLIB_INCLUDE_DIR} ${PARLAYANN_ROOT}/parlaylib ${PARLAYANN_ROOT}/parlaylib/include ${PARLAYANN_ROOT}/third_party/parlaylib)

  if (NOT PARLAYLIB_HEADER_PATH)
    message(FATAL_ERROR "Could not find parlay headers. Set PARLAYLIB_INCLUDE_DIR to the directory containing parlay/parallel.h.")
  endif()

  add_executable(fusedann_parlayann fusedann_parlayann.cpp)

  target_include_directories(fusedann_parlayann PRIVATE
    ${PARLAYANN_ROOT}
    ${PARLAYANN_ROOT}/algorithms
    ${PARLAYLIB_HEADER_PATH})

  target_link_libraries(fusedann_parlayann
    PRIVATE
      OpenMP::OpenMP_CXX
      Threads::Threads)

  if (ENABLE_AVX2)
    target_compile_options(fusedann_parlayann PRIVATE -mavx2 -mfma)
  endif()
endif()
