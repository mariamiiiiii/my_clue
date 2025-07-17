cmake_minimum_required(VERSION 3.8)

project(
  CLUE
  VERSION 1.1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Options: GPU Backend
# ------------------------------------------------------------------------------

set(GPU_BACKEND "CUDA" CACHE STRING "GPU backend to use (CUDA|HIP|NONE)")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS CUDA HIP NONE)
string(TOUPPER "${GPU_BACKEND}" GPU_BACKEND)

# ------------------------------------------------------------------------------
# Alpaka path
# ------------------------------------------------------------------------------

set(ALPAKA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/alpaka CACHE PATH "Path to ALPAKA (by default: submodule)")

# ------------------------------------------------------------------------------
# Verbose makefile
# ------------------------------------------------------------------------------

set(CMAKE_VERBOSE_MAKEFILE OFF)

# ------------------------------------------------------------------------------
# GPU Backend Setup
# ------------------------------------------------------------------------------

include(CheckLanguage)
set(GPU_LANG "")

if(GPU_BACKEND STREQUAL "CUDA")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(GPU_LANG CUDA)
    set(CMAKE_CUDA_STANDARD 17)
#   set(CMAKE_CUDA_ARCHITECTURES "60;70;75")
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  else()
    message(WARNING "CUDA backend selected but CUDA compiler not found.")
  endif()

elseif(GPU_BACKEND STREQUAL "HIP")
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(GPU_LANG HIP)
    set(CMAKE_HIP_STANDARD 17)
    message(STATUS "HIP compiler found: ${CMAKE_HIP_COMPILER}")
  else()
    message(WARNING "HIP backend selected but HIP compiler not found.")
  endif()

else()
  message(STATUS "GPU_BACKEND=NONE selected — building CPU-only version.")
endif()

# ------------------------------------------------------------------------------
# Install prefix
# ------------------------------------------------------------------------------

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# C++ standard check
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if(NOT CMAKE_CXX_STANDARD MATCHES "17|20")
  message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

# ------------------------------------------------------------------------------
# Boost
# ------------------------------------------------------------------------------

set(Boost_DEBUG 1)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

if(Boost_FOUND)
  message(STATUS "Boost package found!")
endif()

# ------------------------------------------------------------------------------
# TBB
# ------------------------------------------------------------------------------

find_package(TBB REQUIRED)
if(TBB_FOUND)
  message(STATUS "TBB package found!")
endif()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

add_subdirectory(clueLib)
add_subdirectory(src)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

enable_testing()

function(run_test_cpu target label repeats)
  FILE(GLOB inputs "${CMAKE_CURRENT_SOURCE_DIR}/data/input/*.csv")
  foreach(input_file IN LISTS inputs)
    add_test(NAME ${input_file}_${label}_CPU COMMAND ${target} -i ${input_file} -d 7.0 -r10.0 -o 2 -e ${repeats} -v)
  endforeach()
endfunction()

function(run_test_gpu target label repeats)
  FILE(GLOB inputs "${CMAKE_CURRENT_SOURCE_DIR}/data/input/*.csv")
  foreach(input_file IN LISTS inputs)
    add_test(NAME ${input_file}_${label}_GPU COMMAND ${target} -i ${input_file} -d 7.0 -r10.0 -o 2 -e ${repeats} -v -u)
  endforeach()
endfunction()

if(GPU_LANG STREQUAL "CUDA")
  run_test_cpu(./src/clue/main main 4)
  run_test_cpu(./src/clue_cuda_alpaka/mainAlpakaCUDA Alpaka 4)
  run_test_gpu(./src/clue/main CUDA 100)
  run_test_gpu(./src/clue_cuda_alpaka/mainAlpakaCUDA AlpakaCUDA 100)

elseif(GPU_LANG STREQUAL "HIP")
  run_test_cpu(./src/clue/main main 4)
  run_test_cpu(./src/clue_hip_alpaka/mainAlpakaHIP Alpaka 4)
  run_test_gpu(./src/clue/main HIP 100)
  run_test_gpu(./src/clue_hip_alpaka/mainAlpakaHIP AlpakaHIP 100)

else()
  run_test_cpu(./src/clue_tbb_alpaka/mainAlpakaTBB TBB 4)
endif()

# ------------------------------------------------------------------------------
# Version + install config
# ------------------------------------------------------------------------------

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLUEVersion.h
               ${CMAKE_CURRENT_BINARY_DIR}/CLUEVersion.h)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CLUEVersion.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CLUE)

include(cmake/CLUECreateConfig.cmake)

install(EXPORT CLUETargets
  NAMESPACE ${PROJECT_NAME}::
  FILE "${PROJECT_NAME}Targets.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/"
)

message(STATUS "Finished!")
