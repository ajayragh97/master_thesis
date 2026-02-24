cmake_minimum_required(VERSION 3.10)
project(RadarVelocityEstimator)

# Use C++14 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})
add_library(radar_estimator_lib 
    src/Estimator.cpp
    src/BodyFrameCorrector.cpp)

target_include_directories(radar_estimator_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
add_executable(run_estimator src/Main.cpp)
target_link_libraries(run_estimator radar_estimator_lib Eigen3::Eigen stdc++fs)
