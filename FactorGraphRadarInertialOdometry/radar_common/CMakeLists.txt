cmake_minimum_required(VERSION 3.10)

add_library(radar_common
    src/Utils.cpp
)

# Allow anyone linking this to see "include/radar_common"
target_include_directories(radar_common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find GTSAM
find_package(GTSAM REQUIRED)
message(STATUS "Found GTSAM version: ${GTSAM_VERSION}")

target_link_libraries(radar_common PUBLIC 
    radar_registration 
    Eigen3::Eigen 
    nlohmann_json::nlohmann_json
    gtsam
)