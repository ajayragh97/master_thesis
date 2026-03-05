cmake_minimum_required(VERSION 3.10)
project(FactorGraphRadarInertialOdometry)

if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected at: $ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global settings for output (puts all binaries in one bin folder)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find External Dependencies once
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include 3rd party headers globally (Nanoflann)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/nanoflann/include)
include_directories(${GTSAM_INCLUDE_DIR})

link_directories(${GTSAM_LIBRARY_DIRS}) 

# Add Sub-directories (Order matters! common first)
add_subdirectory(radar_common)
add_subdirectory(radar_velocity_estimation)
add_subdirectory(radar_registration)
add_subdirectory(factor_graph_optimization)
add_subdirectory(apps)