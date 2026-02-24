#pragma once
#include <Eigen/Dense>
#include <vector>
#include "radar_common/Types.h"

namespace radar
{
    namespace registration
    {
        using namespace radar::common;

        struct ICPConfig
        {
            int max_iterations = 50;
            double max_correspondences_dist = 1.0; // meters
            double intensity_threshold = 0.25; // normalized difference
            double transformation_epsilon = 1e-6; // icp convergence criteria
            double euclidean_fitness_epsilon = 1e-6;
            double start_tx = -0.0465733426578; // init global pose x
            double start_ty = -0.0340903515794; // y
            double start_tz = -0.0261421280906; // z
            double start_qx = -0.0246014325265; // quaternion x
            double start_qy = -0.00101571297063; // quaternion y
            double start_qz = -0.00310958606125; // quaternion z
            double start_qw = 0.999691986724; //quaternion w
        };

        class RadarICP
        {
            public:
                RadarICP(const ICPConfig& config);

                /*
                @brief Computes the transformation from Source to Target cloud
                @param source: The point cloud to be transformed (current frame (time t))
                @param target: The reference point cloud (prev frame (time t-1))
                @param initial_guess: Initial estimate for transformation matrix T
                @return 4x4 Transformation Matrix (Source -> Target)
                */
               Eigen::Matrix4d compute(const PointCloud& source,
                                       const PointCloud& target,
                                       const Eigen::Matrix4d& initial_guess = Eigen::Matrix4d::Identity(),
                                       const bool& perform_2d_icp=true);
                
                /*
                @brief get final alignment error (weighted MSE)
                */
               double getFitnessScore() const { return fitness_score_; }

            private:
                ICPConfig config_;
                double fitness_score_;

                /*
                @brief Normalize intensities of the cloud to [0, 1] range
                @param input pointcloud 
                @return vector of normalized intensities
                */
                std::vector<double> normalizeIntensities(const PointCloud& cloud);

               /*
               @brief Perform weighted SVD
               @param src_pts: Source pointcloud points
               @param tgt_pts: Target pointcloud points
               @param weights: vector of weights for each point
               @return 4x4 Transformation matrix
               */
                Eigen::Matrix4d solveWeightedSVD(const std::vector<Eigen::Vector3d>& src_pts,
                                               const std::vector<Eigen::Vector3d>& tgt_pts,
                                               const std::vector<double>& weights);
                
                Eigen::Matrix4d solveWeightedSVD2D(const std::vector<Eigen::Vector3d>& src_pts,
                                               const std::vector<Eigen::Vector3d>& tgt_pts,
                                               const std::vector<double>& weights);
        };
    }
}