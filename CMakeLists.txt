cmake_minimum_required(VERSION 3.20)
project(NBodySim LANGUAGES CXX)

option(ENABLE_OPENMP "Enable OpenMP backend" ON)
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(PRECISION_FLOAT "Use float precision (default double)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (PRECISION_FLOAT)
  add_compile_definitions(NB_FLOAT_PRECISION)
endif()

# Default release flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src)
