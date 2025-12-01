# Build library with core and backends, then main executable

set(SRC_CORE
  core/integrator.cpp
  core/io.cpp
  core/timing.cpp
)

set(HDR_CORE
  ${CMAKE_SOURCE_DIR}/include/nbody_common.hpp
  ${CMAKE_SOURCE_DIR}/include/integrator.hpp
  ${CMAKE_SOURCE_DIR}/include/io.hpp
  ${CMAKE_SOURCE_DIR}/include/timing.hpp
)

add_library(nbody_core ${SRC_CORE} ${HDR_CORE})
target_include_directories(nbody_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(nbody_core PUBLIC ENABLE_OMP)
  endif()
endif()

add_library(nbody_seq backends/backend_seq.cpp)
target_link_libraries(nbody_seq PUBLIC nbody_core)
target_include_directories(nbody_seq PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(nbody_omp backends/backend_omp.cpp)
target_link_libraries(nbody_omp PUBLIC nbody_core)
target_include_directories(nbody_omp PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(nbody_cuda backends/backend_cuda.cu)
  target_link_libraries(nbody_cuda PUBLIC nbody_core CUDA::cudart)
  target_include_directories(nbody_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
  set_target_properties(nbody_cuda PROPERTIES 
    CUDA_RUNTIME_LIBRARY Shared)
endif()

add_executable(nbody_app app/main.cpp)
target_link_libraries(nbody_app PRIVATE nbody_core nbody_seq nbody_omp)
if (ENABLE_CUDA)
  target_link_libraries(nbody_app PRIVATE nbody_cuda)
  target_compile_definitions(nbody_app PRIVATE ENABLE_CUDA)
endif()
target_include_directories(nbody_app PRIVATE ${CMAKE_SOURCE_DIR}/include)
