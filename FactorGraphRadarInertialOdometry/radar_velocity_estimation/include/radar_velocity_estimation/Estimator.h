#pragma once

#include "radar_common/Types.h"
#include <vector>
#include <random> 

namespace radar
{
    namespace velocity
    {
        using namespace radar::common;

        class RadarEgoEstimator
        {
            public:
                explicit RadarEgoEstimator(const EstimatorConfig& config);
                VelocityEstimate estimate(const PointCloud& raw_scan, const bool& filter_points);
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
}