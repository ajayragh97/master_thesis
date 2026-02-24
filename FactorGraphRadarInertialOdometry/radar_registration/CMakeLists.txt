cmake_minimum_required(VERSION 3.10)

add_library(radar_registration
    src/RadarICP.cpp
)

target_include_directories(radar_registration PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Link Common (for Types) and Velocity (if you use it for pre-filtering inside)
target_link_libraries(radar_registration PUBLIC radar_common radar_velocity_estimation)