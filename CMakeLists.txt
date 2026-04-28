cmake_minimum_required(VERSION 3.14)
project(graphann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Optimization flags: Added -mavx2 and -mfma explicitly
# We also use -fopenmp to ensure the compiler knows to handle parallel pragmas
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -fopenmp -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Library with shared sources
add_library(graphann_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_index.cpp
)
target_include_directories(graphann_lib PUBLIC include)

# Ensure the library itself links to OpenMP
target_link_libraries(graphann_lib PUBLIC OpenMP::OpenMP_CXX)

# Build index executable
add_executable(build_index src/build_index.cpp)
target_link_libraries(build_index PRIVATE graphann_lib OpenMP::OpenMP_CXX)

# Search index executable
add_executable(search_index src/search_index.cpp)
target_link_libraries(search_index PRIVATE graphann_lib OpenMP::OpenMP_CXX)

if(WIN32)
    target_link_libraries(search_index PRIVATE psapi)
endif()