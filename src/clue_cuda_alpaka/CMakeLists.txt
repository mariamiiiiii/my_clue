# mainAlpakaCUDA: this will build the native C++ implementation of CLUE and its
#       corresponding CUDA one built using ALPAKA. Which one to use must be
#       selected at runtime via a flag.

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CUDA)
  add_executable(mainAlpakaCUDA ${PROJECT_SOURCE_DIR}/src/main.cc)
  target_compile_definitions(mainAlpakaCUDA PRIVATE USE_ALPAKA FOR_CUDA ALPAKA_ACC_GPU_CUDA_ENABLED=1)
  target_compile_options(mainAlpakaCUDA PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler;
                        -m64;
                        -expt-relaxed-constexpr;
                        -w;
  >
  )
  target_link_libraries(mainAlpakaCUDA PRIVATE CLUELib)

  target_include_directories(mainAlpakaCUDA PUBLIC
    $<BUILD_INTERFACE:${ALPAKA_DIR}/include>
    PUBLIC
    $<BUILD_INTERFACE:${ALPAKA_DIR}/alpaka/include>
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  )
endif(CMAKE_CUDA_COMPILER)
