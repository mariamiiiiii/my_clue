## Make an automatic library - will be static or dynamic based on user setting
add_library(CLUELib src/CLUEAlgo.cc)

set_target_properties(CLUELib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# We need this directory, and users of our library will need it too
target_include_directories(CLUELib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clueLib/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


file(GLOB _cluelib_headers ${PROJECT_SOURCE_DIR}/clueLib/include/*.h)
set_target_properties(CLUELib PROPERTIES PUBLIC_HEADER "${_cluelib_headers}")

install(TARGETS CLUELib ${INSTALL_LIBRARIES}
  EXPORT CLUETargets
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/CLUE"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

if(CMAKE_CUDA_COMPILER)
  add_library(CLUELibGPU src/CLUEAlgoGPU.cu)

  target_compile_options(CLUELibGPU PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler;
                        -m64;
                        -expt-relaxed-constexpr;
                        -w;
  >
  )

  target_include_directories(CLUELibGPU PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clueLib/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  install(TARGETS CLUELibGPU ${INSTALL_LIBRARIES}
    EXPORT CLUETargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif(CMAKE_CUDA_COMPILER)

