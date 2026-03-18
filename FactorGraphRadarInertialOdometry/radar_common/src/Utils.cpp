#include "radar_common/Types.h"
#include "radar_common/SystemConfig.h"
#include "radar_common/Utils.h"

// Just a dummy function to ensure the object file is created
namespace radar 
{
    namespace common 
    {
        void computePolarAndDirection(RadarTarget& target) 
        {
            const double x = target.position_cart.x();
            const double y = target.position_cart.y();
            const double z = target.position_cart.z();
            target.position_sph[0] = target.position_cart.norm();
            if (target.position_sph[0] < 0.0001) {
                target.position_sph[1] = 0.0; target.position_sph[2] = 0.0;
                target.direction_vector = Eigen::Vector3d::Zero(); return;
            }
            target.direction_vector = target.position_cart / target.position_sph[0];
            target.position_sph[1] = std::atan2(y, x);
            double horizontal_dist = std::sqrt(x*x + y*y);
            target.position_sph[2] = std::atan2(z, horizontal_dist);
        }

        PointCloud loadPointCloud(const std::string& filename) 
        {
            PointCloud cloud;
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) return cloud;
            file.seekg(0, std::ios::end); size_t file_size = file.tellg(); file.seekg(0, std::ios::beg);
            size_t num_floats = file_size / sizeof(float); size_t num_points = num_floats / 5;
            std::vector<float> buffer(num_floats);
            if (file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
                cloud.reserve(num_points);
                for (size_t i = 0; i < num_points; ++i) {
                    RadarTarget t; size_t base = i * 5;
                    t.position_cart = Eigen::Vector3d(buffer[base + 0], buffer[base + 1], buffer[base + 2]);
                    t.snr_db = static_cast<double>(buffer[base + 3]);
                    t.doppler_velocity = static_cast<double>(buffer[base + 4]);
                    computePolarAndDirection(t); cloud.push_back(t);
                }
            }
            return cloud;
        }

        std::vector<double> loadTimestamps(const std::string& path) 
        {
            std::vector<double> timestamps;
            std::ifstream file(path); std::string line;
            while(std::getline(file, line)) if (!line.empty()) timestamps.push_back(std::stod(line));
            return timestamps;
        }

        PointCloud filterDynamicPoints(const PointCloud& input, const Eigen::Vector3d& ego_vel, double threshold) 
        {
            PointCloud static_cloud; static_cloud.reserve(input.size());
            for(const auto& p : input) {
                double expected = -ego_vel.dot(p.direction_vector);
                if(std::abs(p.doppler_velocity - expected) < threshold) static_cloud.push_back(p);
            }
            return static_cloud;
        }

        std::vector<ImuData> loadImuDataVector(const std::string& data_path, const std::string& time_path)
        {
            std::vector<ImuData> imu_vec;
            std::ifstream f_data(data_path);
            std::ifstream f_time(time_path);

            if (!f_data.is_open() || !f_time.is_open()) return imu_vec;

            double t, ax, ay, az, gx, gy, gz;
            while (f_time >> t)
            {
                if (!(f_data >> ax >> ay >> az >> gx >> gy >> gz)) break;
                
                ImuData data;
                data.timestamp = t;
                data.linear_acceleration = Eigen::Vector3d(ax, ay, az);
                data.angular_velocity = Eigen::Vector3d(gx, gy, gz);
                imu_vec.push_back(data);
            }
            return imu_vec;
        }

        std::vector<GTData> loadGroundTruth(const std::string& pose_file, const std::string& time_file)
        {
            std::vector<GTData> gt_list;
            std::ifstream f_pose(pose_file), f_time(time_file);
            double t, x, y, z, qx, qy, qz, qw;
            while (f_time >> t && f_pose >> x >> y >> z >> qx >> qy >> qz >> qw) 
            {
                gt_list.push_back({t, gtsam::Pose3(gtsam::Rot3::Quaternion(qw, qx, qy, qz), gtsam::Point3(x, y, z))});
            }
            return gt_list;
        }

        gtsam::Pose3 getClosestGTPose(const std::vector<GTData>& gt_list, double target_time) 
        {
            auto it = std::min_element(gt_list.begin(), gt_list.end(),
                [target_time](const GTData& a, const GTData& b) { return std::abs(a.timestamp - target_time) < std::abs(b.timestamp - target_time); });
            return it->pose;
        }

        int getClosestRadarIdx(const std::vector<double>& timestamps, double target_time) 
        {
            auto it = std::min_element(timestamps.begin(), timestamps.end(),
                [target_time](double a, double b) { return std::abs(a - target_time) < std::abs(b - target_time); });
            return std::distance(timestamps.begin(), it);
        }

        ImuBias getImuBias(const std::vector<ImuData>& all_imu_data, const double& static_duration)
        {
            ImuBias imu_bias;
            
            // Safety check
            if (all_imu_data.empty()) return imu_bias;

            int static_count = 0;
            std::vector<double> delta_t_imu;

            // ==========================================
            // 1. Calculate Mean (Bias) and dt
            // ==========================================
            for (const auto& imu : all_imu_data) 
            {
                if (imu.timestamp - all_imu_data[0].timestamp < static_duration) 
                {
                    imu_bias.bias_acc += imu.linear_acceleration;
                    imu_bias.bias_gyro += imu.angular_velocity;

                    // Look backward to calculate dt to avoid out-of-bounds
                    if (static_count > 0)
                    {
                        delta_t_imu.push_back(imu.timestamp - all_imu_data[static_count - 1].timestamp);
                    }
                    static_count++;
                } 
                else break;
            }  

            if (static_count > 0) 
            { 
                imu_bias.bias_acc /= static_count; 
                imu_bias.bias_gyro /= static_count; 
                
                // Divide by the actual size of the delta array
                if (!delta_t_imu.empty()) {
                    double sum_time_diff = std::accumulate(delta_t_imu.begin(), delta_t_imu.end(), 0.0);
                    imu_bias.mean_imu_dt = sum_time_diff / delta_t_imu.size();
                }
            }

            // ==========================================
            // 2. Calculate Standard Deviations
            // ==========================================
            for (const auto& imu : all_imu_data) 
            {
                if (imu.timestamp - all_imu_data[0].timestamp < static_duration)
                {
                    // Use .square() and convert back to .matrix() before accumulating
                    imu_bias.std_acc += (imu.linear_acceleration - imu_bias.bias_acc).array().square().matrix();
                    imu_bias.std_gyro += (imu.angular_velocity - imu_bias.bias_gyro).array().square().matrix();
                }
                else break;
            }
            
            // Divide by (N - 1) for sample standard deviation
            if (static_count > 1)
            {
                imu_bias.std_acc = (imu_bias.std_acc / (static_count - 1)).cwiseSqrt();
                imu_bias.std_gyro = (imu_bias.std_gyro / (static_count - 1)).cwiseSqrt();
            }
            else 
            {
                // Fallback if only 1 frame was provided
                imu_bias.std_acc = Eigen::Vector3d::Zero();
                imu_bias.std_gyro = Eigen::Vector3d::Zero();
            }

            return imu_bias;
        }
    }
}