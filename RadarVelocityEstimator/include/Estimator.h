#pragma once

#include "Types.h"
#include <vector>
#include <random> 

namespace radar_estimator
{
    class RadarEgoEstimator
    {
        public:
            explicit RadarEgoEstimator(const EstimatorConfig& config);
            VelocityEstimate estimate(const PointCloud& raw_scan);
            void updateConfig(const EstimatorConfig& config);

        private:
            EstimatorConfig config_;
            std::mt19937 rng_;
            PointCloud preprocess(const PointCloud& raw_scan);
            bool isStationary(const PointCloud& scan);
            std::vector<int> performRANSAC(const PointCloud& scan);
            VelocityEstimate SolveLeastSquares(const PointCloud& scan,
                                               const std::vector<int>& inliers);
    };
}