add_library(radar_velocity_estimation
    src/Estimator.cpp
    src/BodyFrameCorrector.cpp
)

target_include_directories(radar_velocity_estimation PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# LINK TO COMMON 
target_link_libraries(radar_velocity_estimation PUBLIC radar_common)