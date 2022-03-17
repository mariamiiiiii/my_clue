# Native C++ CLUE and TBB CLUE using CUPLA
add_subdirectory(clue_tbb_cupla)

if(CMAKE_CUDA_COMPILER)
  # Native C++ CLUE and native CUDA CLUE
  add_subdirectory(clue)

  # Native C++ CLUE and CUDA CLUE using CUPLA
  add_subdirectory(clue_cuda_cupla)
endif(CMAKE_CUDA_COMPILER)
