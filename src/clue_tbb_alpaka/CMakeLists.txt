# mainAlpakaTBB: this will build the native C++ implementation of CLUE and its
#       corresponding TBB one built using ALPAKA. Which one to use must be
#       selected at runtime via a flag.

set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CXX)
add_executable(mainAlpakaTBB ${PROJECT_SOURCE_DIR}/src/main.cc)
target_compile_definitions(mainAlpakaTBB PRIVATE USE_ALPAKA ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
target_link_libraries(mainAlpakaTBB PRIVATE CLUELib tbb)

target_include_directories(mainAlpakaTBB PUBLIC
  $<BUILD_INTERFACE:${ALPAKA_DIR}/include>
  PUBLIC
  $<BUILD_INTERFACE:${ALPAKA_DIR}/alpaka/include>
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
