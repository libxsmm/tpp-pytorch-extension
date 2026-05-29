# cmake version
cmake_minimum_required(VERSION 3.18)

# project name
project(turboquant VERSION 0.1.0)

# binary paths to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(turboquant 
    src/main.cpp
    src/init.cpp
    src/rotation.cpp
)

# set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


target_include_directories(turboquant 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libxsmm/include
)

#link mkl libraries
target_link_libraries(turboquant 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libxsmm/lib/libxsmm.a
    iomp5
    mkl_rt
)

# add compiler options for debug and release builds
target_compile_options(turboquant 
    PUBLIC 
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O2 -march=native>
)

# set output directory for DEBUG and RELEASE builds
set_target_properties(turboquant PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)


# ------------------------------------------------- Testing Phase -------------------------------------------------

# Add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()


# Test executable
add_executable(turboquant_tests
    tests/QR_test.cpp
    src/init.cpp
    src/rotation.cpp
)

target_include_directories(turboquant_tests 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libxsmm/include
)

target_link_libraries(turboquant_tests 
    PRIVATE
    gtest_main
    gtest
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libxsmm/lib/libxsmm.a
    iomp5
    mkl_rt
)

target_compile_options(turboquant_tests 
    PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O2 -march=native>
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(turboquant_tests)