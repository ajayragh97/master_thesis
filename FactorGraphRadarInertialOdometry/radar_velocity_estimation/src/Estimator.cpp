#include "radar_velocity_estimation/Estimator.h"
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace radar
{
    namespace velocity
    {

        RadarEgoEstimator::RadarEgoEstimator(const EstimatorConfig& config)   :   config_(config) 
        {
            std::random_device rd;
            rng_ = std::mt19937(rd());
        }

        // update the config with new config parameters
        void RadarEgoEstimator::updateConfig(const EstimatorConfig& config)
        {
            config_ = config;
        }
        
        // preprocess the raw scan to remove noise
        PointCloud RadarEgoEstimator::preprocess(const PointCloud& raw_scan)
        {
            PointCloud clean_points;
            clean_points.reserve(raw_scan.size());

            double min_elv_rad = config_.min_elv * M_PI / 180.0;
            double max_elv_rad = config_.max_elv * M_PI / 180.0;
            double min_azim_rad = config_.min_azim * M_PI / 180.0;
            double max_azim_rad = config_.max_azim * M_PI / 180.0;

            for (const auto& p : raw_scan)
            {
                if (p.position_sph[0] < config_.min_dist || p.position_sph[0] > config_.max_dist) 
                {
                    std::cout << "out of range" << std::endl;
                    continue;
                }
                if (p.position_sph[1] < min_azim_rad || p.position_sph[1] > max_azim_rad) 
                {
                    std::cout << "outside azimuth window" << std::endl;
                    continue;
                }
                if (p.position_sph[2] < min_elv_rad || p.position_sph[2] > max_elv_rad) 
                {
                    std::cout << "Outside elevation window" << std::endl;
                    continue;
                }
                // if (p.snr_db < config_.min_db)
                // {
                //     std::cout << "too low snr: " << p.snr_db << std::endl;
                //     continue;
                // } 

                clean_points.push_back(p);
            }

            return clean_points;
        }

        // check whether the system is stationary or not by checking mean doppler velocity
        bool RadarEgoEstimator::isStationary(const PointCloud& scan)
        {
            // checking median doppler velocities
            if(scan.empty()) return true;

            std::vector<double> dopplers;
            dopplers.reserve(scan.size());

            for (const RadarTarget& p : scan)
            {
                dopplers.push_back(std::abs(p.doppler_velocity));
            }

            size_t n = dopplers.size() / 2;
            std::nth_element(dopplers.begin(), dopplers.begin() + n, dopplers.end());
            double median_doppler = dopplers[n];
            return median_doppler < config_.zero_vel_thresh;

        }

        VelocityEstimate RadarEgoEstimator::SolveLeastSquares(const PointCloud& scan,
                                                            const std::vector<int>& inliers)
        {
            VelocityEstimate res;
            int n = inliers.size();

            // min 3 points required to solve for linear velocity
            if (n < 3)
            {
                res.success = false;
                return res;
            }

            // matrix H (n x 3), vector y (n x 1) 
            Eigen::MatrixXd H(n, 3);
            Eigen::VectorXd y(n);

            for (int i = 0; i < n; ++i)
            {
                const RadarTarget& target = scan[inliers[i]];
                
                H(i, 0) = -target.direction_vector[0];
                H(i, 1) = -target.direction_vector[1];
                H(i, 2) = -target.direction_vector[2];

                y(i) = target.doppler_velocity;
            }

            // solve the system of equations Hx = y to get x
            res.linear_velocity = H.colPivHouseholderQr().solve(y);
            res.success = true;

            // covariance estimation
            Eigen::VectorXd residuals = y - H * res.linear_velocity;
            double mse = residuals.squaredNorm() / std::max(1, (n - 3));
            res.covariance = (H.transpose() * H).inverse() * mse;
            res.inlier_ratio = ((n * 1.0) / scan.size()) * 100;
            // std::cout << "inlier count: " << n << std::endl;
            // std::cout << "scan count: " << scan.size() << std::endl;
            // std::cout << "ratio: " << ((n * 1.0) / scan.size()) * 100 << std::endl;
            
            return res;

        }

        std::vector<int> RadarEgoEstimator::performRANSAC(const PointCloud& scan)
        {
            int n_points = scan.size();
            if (n_points < 3) return {}; //minimum 3 points required to solve for velocity

            std::vector<int> best_inliers;
            int max_inlier_count = 0;

            std::uniform_int_distribution<int> dist(0, n_points - 1);

            for (int i = 0; i < config_.ransac_iter; ++i)
            {
                std::vector<int> sample_indices;
                while (sample_indices.size() < 3)
                {
                    int idx = dist(rng_);
                    
                    // check if the idx is unique
                    if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end())
                    {
                        sample_indices.push_back(idx);
                    }
                }

                // solve for candidate velocity using these 3 points
                VelocityEstimate candidate = SolveLeastSquares(scan, sample_indices); 
                if (!candidate.success) continue;

                // check consensus
                std::vector<int> current_inliers;
                for (int j = 0; j < n_points; ++j)
                {
                    const RadarTarget& p = scan[j];
                    
                    // predicted doppler
                    double predicted_doppler = -candidate.linear_velocity.dot(p.direction_vector);

                    // predicted vs measured
                    if (std::abs(p.doppler_velocity - predicted_doppler) < config_.inlier_thresh)
                    {
                        current_inliers.push_back(j);
                    }
                }

                if ((int)current_inliers.size() > max_inlier_count)
                {
                    max_inlier_count = current_inliers.size();
                    best_inliers = current_inliers;
                }

                if (max_inlier_count > n_points * config_.max_inlier_ratio) break;
            }

            return best_inliers;
        }

        VelocityEstimate RadarEgoEstimator::estimate(const PointCloud& raw_scan, const bool& filter_points)
        {
            // std::cout << "Input cloud size: " << raw_scan.size() << std::endl;
            VelocityEstimate result;
            PointCloud filtered_cloud;

            // preprocess
            if (filter_points)
            {
                filtered_cloud = preprocess(raw_scan);
            }
            else
            {
                filtered_cloud = raw_scan;
            }
            
            if (filtered_cloud.size() < 3) // atleast 3 points are required to solve
            {
                result.success = false;
                return result;
            }

            if(isStationary(filtered_cloud))
            {
                // std::cout << "scan is stationary" << std::endl;
                result.success = true;
                result.linear_velocity = Eigen::Vector3d::Zero();
                result.covariance = Eigen::Matrix3d::Identity() * 0.0001;
                return result;
            }

            std::vector<int> inlier_indices = performRANSAC(filtered_cloud);

            // std::cout << "inliers after RANSAC: " << inlier_indices.size() << std::endl;
            result = SolveLeastSquares(filtered_cloud, inlier_indices);
            return result;        
        }
    }
}