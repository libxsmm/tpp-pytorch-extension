# cmake version
cmake_minimum_required(VERSION 3.18)

# project name
project(turboquant VERSION 0.1.0)

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

# # link to oneAPI present at path: /opt/intel/oneapi/compiler/latest/linux/lib
# target_link_libraries(turboquant 
#     PUBLIC 
#     /data/swtools/intel/2025.2.0/2025.2/lib/libiomp5.so
# )




