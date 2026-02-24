#pragma once
#include "radar_common/Types.h"
#include <vector>
#include <string>
#include <map>

namespace radar
{
    namespace velocity
    {
        using namespace radar::common;
        
        class BodyFrameCorrector
        {
            public:
                BodyFrameCorrector();

                /*
                @brief Loads the transform parameters from txt file
                Required data format:
                Line 1: x y z
                Line 2: qx qy qz qw
                @param string: filepath
                @param Transform: out_transform
                */
                bool loadTransform(const std::string& filepath, Transform& out_transform);

                /*
                @brief Load IMU data from txt file
                @param string: imu data file path
                @param string: timestamp file path
                */
                bool loadImuData(const std::string& data_path, const std::string& time_path);

                /*
                @brief sets transforms manually
                @param Transform: base to radar transform
                @param Transform: base to IMU transform
                */
                void setTransform(const Transform& base_to_radar, const Transform& base_to_imu);

                /*
                @brief Corrects the velocity by transforming from sensor frame to body frame 

                Math: V_base = R * V_radar - (omega_base * lever_arm)

                @param Eigen::Vector3d: velocity estimated in Radar frame
                @param double: Timestamp of the estimated velocity (radar scan timestamp)
                @return Eigen::Vector3d Velocity in Base Frame
                */
                Eigen::Vector3d correctVelocity(const Eigen::Vector3d& radar_velocity, double timestamp);
                
                Eigen::Matrix4d getRadarToBaseTransform() const;
            private:
                /*
                @brief calculate angular velocity at timestamp via linear interpolation
                */
                Eigen::Vector3d getInterpolatedOmega(double timestamp);

            // Calibration transforms
            Transform T_base_radar;
            Transform T_base_imu;

            // IMU data Buffer(Timestamp -> Data)
            std::map<double, ImuData> imu_buffer;
        };
    }
}