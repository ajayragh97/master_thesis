#include "radar_velocity_estimation/BodyFrameCorrector.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace radar
{
    namespace velocity
    {

        BodyFrameCorrector::BodyFrameCorrector()
        {
            // Initialize the system callibration transforms
            T_base_radar.translation.setZero();
            T_base_radar.rotation.setIdentity();
            T_base_imu.translation.setZero();
            T_base_imu.rotation.setIdentity();

        }

        void BodyFrameCorrector::setTransform(const Transform& base_to_radar, const Transform& base_to_imu)
        {
            T_base_radar = base_to_radar;
            T_base_imu = base_to_imu;
        }

        bool BodyFrameCorrector::loadTransform(const std::string& filepath, Transform& out_transform)
        {
            std::ifstream file(filepath);
            if (!file.is_open())
            {
                std::cerr << "Error opening transform file: " << filepath << std::endl;
                return false;
            }

            double tx, ty, tz;
            double qx, qy, qz, qw;

            file >> tx >> ty >> tz;
            file >> qx >> qy >> qz >> qw;

            out_transform.translation = Eigen::Vector3d(tx, ty, tz);
            out_transform.rotation = Eigen::Quaterniond(qw, qx, qy, qz);
            out_transform.rotation.normalize();

            return true;
        }

        bool BodyFrameCorrector::loadImuData(const std::string& data_path, const std::string& time_path)
        {
            std::ifstream f_data(data_path);
            std::ifstream f_time(time_path);

            if (!f_data.is_open() || !f_time.is_open())
            {
                std::cerr << "Error opening IMU files" << std::endl;
                return false;
            }

            double t;
            double ax, ay, az, gx, gy, gz;
            int count = 0;

            while (f_time >> t)
            {
                if (!(f_data >> ax >> ay >> az >> gx >> gy >> gz)) break;

                ImuData data;
                data.timestamp = t;
                data.linear_acceleration = Eigen::Vector3d(ax, ay, az);
                data.angular_velocity = Eigen::Vector3d(gx, gy, gz);

                imu_buffer[t] = data;
                count++;
            }

            std::cout << "Loaded " << count << "IMU Measurements" << std::endl;
            return count > 0;
        }

        Eigen::Matrix4d BodyFrameCorrector::getRadarToBaseTransform() const
        {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = T_base_radar.rotation.toRotationMatrix();
            T.block<3,1>(0,3) = T_base_radar.translation;
            return T;
        }

        Eigen::Vector3d BodyFrameCorrector::getInterpolatedOmega(double timestamp)
        {
            if (imu_buffer.empty()) return Eigen::Vector3d::Zero();

            // lower_bound -> to find first imu data >= timestamp
            auto it_upper = imu_buffer.lower_bound(timestamp);

            // case1 : requested timestamp is before first IMU message
            // return: first IMU message
            if (it_upper == imu_buffer.begin()) return it_upper->second.angular_velocity;

            // case2 : requested timestamp is after last IMU message
            // return: last imu message
            if (it_upper == imu_buffer.end()) return imu_buffer.rbegin()->second.angular_velocity;

            // case3 : interpolate between previous IMU data (it_lower) and returned IMU_data (it_upper)
            // return: interpolated angular velocity
            auto it_lower = std::prev(it_upper);

            double t0 = it_lower->first;
            double t1 = it_upper->first;
            double ratio = (timestamp - t0) / (t1 - t0);

            Eigen::Vector3d w0 = it_lower->second.angular_velocity;
            Eigen::Vector3d w1 = it_upper->second.angular_velocity;

            return w0 + (w1 - w0) * ratio;
        }

        Eigen::Vector3d BodyFrameCorrector::correctVelocity(const Eigen::Vector3d& radar_vel, double timestamp)
        {
            // get angular velocity at timestamp
            Eigen::Vector3d w_imu = getInterpolatedOmega(timestamp);

            // Transform angular velocity from IMU frame to base frame
            // w_base = R_base_imu * w_imu
            Eigen::Vector3d w_base = T_base_imu.rotation * w_imu;

            // Rotate radar linear velocity to base frame
            // V_radar_aligned = R_base_radar * V_radar
            Eigen::Vector3d v_radar_aligned = T_base_radar.rotation * radar_vel;

            // calculate tangential velocity from lever arm effect
            // lever_arm = translation from base to radar
            // tangential velocity = w_base cross(X) lever_arm
            Eigen::Vector3d lever_arm = T_base_radar.translation;
            Eigen::Vector3d v_tangential = w_base.cross(lever_arm);

            // final correaction: v_base = v_radar - v_tangential
            return v_radar_aligned - v_tangential;
        }
    }
}