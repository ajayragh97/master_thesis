#pragma once

#include "radar_common/Types.h"
#include <nanoflann.hpp>

namespace radar
{
    namespace registration
    {
        using namespace radar::common;

        /*
        @brief Adaptor to allow Nanoflann to read radar::common::PointCloud
        */
       struct PointCloudAdapter
       {
            const PointCloud& cloud;

            PointCloudAdapter(const PointCloud& cloud_in) : cloud(cloud_in) { }

            // Interfaces required by nanoflann

            // Return number of points
            inline size_t kdtree_get_point_count() const { return cloud.size(); }
            
            // Return L2 distance between query point and point in cloud
            inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t) const
            {
                const double d0 = p1[0] - cloud[idx_p2].position_cart.x();
                const double d1 = p1[1] - cloud[idx_p2].position_cart.y();
                const double d2 = p1[2] - cloud[idx_p2].position_cart.z();

                return d0*d0 + d1*d1 + d2*d2;
            }

            // Return coordinate of point at index 'idx' in dimension 'dim'
            inline double kdtree_get_pt(const size_t idx, const size_t dim) const
            {
                if (dim == 0) return cloud[idx].position_cart.x();
                if (dim == 1) return cloud[idx].position_cart.y();
                return cloud[idx].position_cart.z();
            }

            template <class BBOX>
            bool kdtree_get_bbox(BBOX& ) const { return false; }
       };

       // Define KD-Tree type
       using RadarKdTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, PointCloudAdapter>,
            PointCloudAdapter,
            3>;
    }
}