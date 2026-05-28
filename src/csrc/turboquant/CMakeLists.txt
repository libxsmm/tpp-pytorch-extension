# cmake version
cmake_minimum_required(VERSION 3.18)

# project name
project(turboquant VERSION 0.1.0)

# binary paths to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(turboquant 
    src/main.cpp
    src/init.cpp
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

target_link_libraries(turboquant 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libxsmm/lib/libxsmm.a
)

# link iomp5 library
target_link_libraries(turboquant 
    PUBLIC 
    iomp5
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



