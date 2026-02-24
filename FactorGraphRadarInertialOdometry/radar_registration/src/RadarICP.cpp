#include "radar_registration/RadarICP.h"
#include "radar_registration/kdTreeAdaptor.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/SVD>

namespace radar 
{
    namespace registration 
    {
        RadarICP::RadarICP(const ICPConfig& config) : config_(config), fitness_score_(0.0) {}

        std::vector<double> RadarICP::normalizeIntensities(const PointCloud& cloud)
        {
            std::vector<double> intensities;
            intensities.reserve(cloud.size());

            if (cloud.empty()) return intensities;

            // load raw snr values in db
            std::vector<double> raw_snr;
            raw_snr.reserve(cloud.size());
            for (const auto& p : cloud)
            {
                raw_snr.push_back(p.snr_db);
            }

            // Find 99th percentile snr measurement
            std::vector<double> sorted_snr = raw_snr;
            size_t n = sorted_snr.size();
            size_t idx_99 = static_cast<size_t>(n * 0.99);
            std::nth_element(sorted_snr.begin(), sorted_snr.begin() + idx_99, sorted_snr.end());
            double max_val = sorted_snr[idx_99];

            // Avoid division by zero
            if (max_val < 1.0) max_val = 1.0;  // Note: SNR values has to be +ve for this

            //Normalize
            for (double val : raw_snr)
            {
                double norm = val / max_val;
                if (norm > 1.0) norm = 1.0;
                if (norm < 0.0) norm = 0.0;
                intensities.push_back(norm);
            }

            return intensities;         
        }

        Eigen::Matrix4d RadarICP::solveWeightedSVD(const std::vector<Eigen::Vector3d>& src_pts,
                                         const std::vector<Eigen::Vector3d>& tgt_pts,
                                         const std::vector<double>& weights)
        {
            if (src_pts.empty()) return Eigen::Matrix4d::Identity();

            // Compute weighted centroids
            Eigen::Vector3d c_src = Eigen::Vector3d::Zero();
            Eigen::Vector3d c_tgt = Eigen::Vector3d::Zero();
            double sum_w = 0.0;

            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                c_src += weights[i] * src_pts[i];
                c_tgt += weights[i] * tgt_pts[i];
                sum_w += weights[i];
            }

            if (sum_w < 1e-6) return Eigen::Matrix4d::Identity(); // Avoid division by zero

            c_src /= sum_w;
            c_tgt /= sum_w;

            // Compute weighted covariance matrix
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                Eigen::Vector3d p = src_pts[i] - c_src;
                Eigen::Vector3d q = tgt_pts[i] - c_tgt;
                H += weights[i] * (p * q.transpose());
            }

            // SVD
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // Transformation
            Eigen::Matrix3d R = V * U.transpose();

            // Reflection check (determinant should be +1)
            if (R.determinant() < 0)
            {
                V.col(2) *= -1;
                R = V * U.transpose();
            }

            Eigen::Vector3d t = c_tgt - (R * c_src);
            
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = R;
            T.block<3,1>(0,3) = t;

            return T;
        }

        Eigen::Matrix4d RadarICP::solveWeightedSVD2D(const std::vector<Eigen::Vector3d>& src_pts,
                                               const std::vector<Eigen::Vector3d>& tgt_pts,
                                               const std::vector<double>& weights)
        {

            if (src_pts.empty()) return Eigen::Matrix4d::Identity();

            // Compute weighted centroids 2D
            Eigen::Vector2d c_src = Eigen::Vector2d::Zero();
            Eigen::Vector2d c_tgt = Eigen::Vector2d::Zero();
            double sum_w = 0.0;

            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                c_src += weights[i] * src_pts[i].head<2>();
                c_tgt += weights[i] * tgt_pts[i].head<2>();
                sum_w += weights[i];
            }

            if (sum_w < 1e-6) return Eigen::Matrix4d::Identity();

            c_src /= sum_w;
            c_tgt /= sum_w;

            // Compute 2x2 weighted covariance matrix
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                Eigen::Vector2d p = src_pts[i].head<2>() - c_src;
                Eigen::Vector2d q = tgt_pts[i].head<2>() - c_tgt;
                H += weights[i] * (p * q.transpose());
            }
            
            // 2D SVD
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d U = svd.matrixU();
            Eigen::Matrix2d V = svd.matrixV();

            // 2D Rotation
            Eigen::Matrix2d R_2d = V * U.transpose();

            // Reflection check for 2D
            if (R_2d.determinant() < 0)
            {
                V.col(1) *= -1;
                R_2d = V * U.transpose();
            }

            // 2D translation
            Eigen::Vector2d t_2d = c_tgt - (R_2d * c_src);

            // 4x4 transformation matrix
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<2,2>(0,0) = R_2d;
            T(0,3) = t_2d.x();
            T(1,3) = t_2d.y();

            return T;
        }

        Eigen::Matrix4d RadarICP::compute(const PointCloud& source,
                                          const PointCloud& target,
                                          const Eigen::Matrix4d& initial_guess,
                                          const bool& perform_2d_icp)
        {
            if (source.empty() || target.empty()) return initial_guess;

            // Normalize intensities
            std::vector<double> src_int = normalizeIntensities(source);
            std::vector<double> tgt_int = normalizeIntensities(target);

            // Build KD-Tree on Target cloud
            PointCloudAdapter adaptor(target);
            RadarKdTree index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            index.buildIndex();

            Eigen::Matrix4d transformation = initial_guess;

            // ICP iteration loop
            for (int iter = 0; iter < config_.max_iterations; ++iter)
            {
                std::vector<Eigen::Vector3d> matched_src;
                std::vector<Eigen::Vector3d> matched_tgt;
                std::vector<double> matched_weights;

                double current_error = 0.0;

                // Transform source point and find correspondences
                for (size_t i = 0; i < source.size(); ++i)
                {
                    Eigen::Vector4d p_hom;
                    p_hom << source[i].position_cart, 1.0;
                    Eigen::Vector3d p_transformed = (transformation * p_hom).head<3>();

                    // Query KD-Tree (Nearest Neighbour)
                    double query_pt[3] = {p_transformed.x(), p_transformed.y(), p_transformed.z()};
                    size_t ret_index;
                    double out_dist_sqr;
                    nanoflann::KNNResultSet<double> resultSet(1);
                    resultSet.init(&ret_index, &out_dist_sqr);
                    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));

                    // Filter Matches
                    // Distance check
                    if (out_dist_sqr > config_.max_correspondences_dist * config_.max_correspondences_dist)
                    {
                        continue;
                    }   

                    // Intensity check
                    double i_s = src_int[i];
                    double i_t = tgt_int[ret_index];

                    if (std::abs(i_s - i_t) > config_.intensity_threshold)
                    {
                        continue;
                    }

                    // store match
                    matched_src.push_back(p_transformed);
                    matched_tgt.push_back(target[ret_index].position_cart);

                    // Weighting
                    double w = std::min(i_s, i_t);
                    matched_weights.push_back(w);

                    current_error += w * out_dist_sqr;
                }

                if (matched_src.size() < 10)
                {
                    std::cerr << "Not enough matches to do ICP" << std::endl;
                    break;
                }

                // Calculate delta transformation
                Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
                if (perform_2d_icp)
                {
                    delta_T = solveWeightedSVD2D(matched_src, matched_tgt, matched_weights);
                }
                else
                {
                    delta_T = solveWeightedSVD2D(matched_src, matched_tgt, matched_weights);
                }

                // Update global transformation T
                transformation = delta_T * transformation;

                // Checking for convergence
                // Check if delta_T is close to Identity
                double translation_mag = delta_T.block<3,1>(0,3).norm();
                double rotation_mag = std::abs(std::acos(std::clamp((delta_T(0,0) +
                                                                     delta_T(1,1) +
                                                                     delta_T(2,2) -1.0)*0.5, -1.0, 1.0)));
                
                if (translation_mag < config_.transformation_epsilon && rotation_mag < config_.transformation_epsilon)
                {
                    // converged
                    break;
                }

                fitness_score_ = current_error / matched_src.size();

            }

            return transformation;
        }
        

    }
}