cmake_minimum_required(VERSION 3.8)

# Location of the cupla
set(cupla_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cupla CACHE PATH "Path to CUPLA (by default: submodule)")

# Location of TBB
set(TBB_DIR "/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/tbb/2020_U2-ghbfee/cmake/TBB/")

project(
  CLUE
  VERSION 1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CXX)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES "60;70;75")
  set(MAX_CUDA_VERSION "11.2.142")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER ${MAX_CUDA_VERSION})
    message(FATAL_ERROR "Max supported NVCC is ${MAX_CUDA_VERSION}")
  endif()
else()
  message(STATUS "No CUDA compiler found. Still, you can run the C++ version!")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

include(GNUInstallDirs)

# Set up C++ Standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if(NOT CMAKE_CXX_STANDARD MATCHES "17|20")
  message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost package found!")
endif()

find_package(TBB REQUIRED)
if(TBB_FOUND)
  message(STATUS "TBB package found!")
endif()

add_subdirectory(clueLib)
add_subdirectory(src)

message(STATUS "Finished!")

