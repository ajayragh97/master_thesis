# Velocity Estimator
add_executable(run_velocity_estimator offline_odometry/Velocity_Estimator.cpp)

# Link requirements
target_link_libraries(run_velocity_estimator 
    PRIVATE 
    radar_velocity_estimation   # The module we are testing
    radar_common     # The types
    Eigen3::Eigen    # Math
    stdc++fs         # Filesystem (needed for C++17 on some compilers)
)


# Ensure it finds headers
target_include_directories(run_velocity_estimator PRIVATE 
    ${CMAKE_SOURCE_DIR}/radar_common/include
    ${CMAKE_SOURCE_DIR}/radar_velocity_estimation/include
)

# Radar Registration
add_executable(run_radar_registration offline_odometry/Radar_Registration.cpp)

target_link_libraries(run_radar_registration 
    PRIVATE 
    radar_registration   
    radar_velocity_estimation  
    radar_common  
    Eigen3::Eigen    
    stdc++fs        
)

target_include_directories(run_radar_registration PRIVATE 
    ${CMAKE_SOURCE_DIR}/radar_common/include
    ${CMAKE_SOURCE_DIR}/radar_velocity_estimation/include
    ${CMAKE_SOURCE_DIR}/radar_registration/include
    ${CMAKE_SOURCE_DIR}/3rdparty/nanoflann/include
)

# Single Frame ICP test
add_executable(test_single_frame_icp single_frame_test/Main.cpp)
target_link_libraries(test_single_frame_icp 
    PRIVATE 
    radar_registration 
    radar_velocity_estimation 
    radar_common 
    Eigen3::Eigen 
    stdc++fs
)
target_include_directories(test_single_frame_icp PRIVATE 
    ${CMAKE_SOURCE_DIR}/radar_common/include
    ${CMAKE_SOURCE_DIR}/radar_velocity_estimation/include
    ${CMAKE_SOURCE_DIR}/radar_registration/include
    ${CMAKE_SOURCE_DIR}/3rdparty/nanoflann/include
)

# Yaw estimator
add_executable(run_yaw_estimator offline_odometry/delta_yaw_estimator.cpp)

target_link_libraries(run_yaw_estimator 
    PRIVATE 
    radar_registration   
    radar_velocity_estimation  
    radar_common  
    Eigen3::Eigen    
    stdc++fs        
)

target_include_directories(run_yaw_estimator PRIVATE 
    ${CMAKE_SOURCE_DIR}/radar_common/include
    ${CMAKE_SOURCE_DIR}/radar_velocity_estimation/include
    ${CMAKE_SOURCE_DIR}/radar_registration/include
    ${CMAKE_SOURCE_DIR}/3rdparty/nanoflann/include
)