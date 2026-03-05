cmake_minimum_required(VERSION 3.10)


add_library(factor_graph_optimization
    src/GraphOptimizer.cpp
)

target_include_directories(factor_graph_optimization PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Find GTSAM
find_package(GTSAM REQUIRED)
message(STATUS "Found GTSAM version: ${GTSAM_VERSION}")

# Link against our radar packages and GTSAM
target_link_libraries(factor_graph_optimization PUBLIC 
    radar_common 
    radar_velocity_estimation
    gtsam
)