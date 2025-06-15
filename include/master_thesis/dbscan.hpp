#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "datastructure.hpp"

namespace clustering
{ 

class DBSCAN_cluster
{
public:
    DBSCAN_cluster(double eps, int minPts) : epsilon(eps), min_points(minPts) {}

    // Main function to run DBSCAN clustering
    void fit(std::vector<datastructures::Point>& points);

private:
    double epsilon;
    int min_points;

    // Euclidean distance calculation
    double distance(const datastructures::Point& p1, const datastructures::Point& p2);

    // find the highest intensity in a given set of points
    float find_intensity_bounds(std::vector<datastructures::Point>& points, float& max_int, float& min_int);

    // Get neighbors of a point
    std::vector<int> regionQuery(const std::vector<datastructures::Point>& points, const datastructures::Point& point, const float& max_int, const float& min_int);

    // Expand the cluster
    bool expandCluster(std::vector<datastructures::Point>& points, int point_idx, int cluster_id);

    

};

// Distance function to calculate the Euclidean distance between two points
double DBSCAN_cluster::distance(const datastructures::Point& p1, const datastructures::Point& p2)
{
    // std::cout << "distance: " << std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2)) ;
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

// Function to get all neighbors within the epsilon distance of a point
std::vector<int> DBSCAN_cluster::regionQuery(const std::vector<datastructures::Point>& points, const datastructures::Point& point, const float& max_int, const float& min_int )
{
    std::vector<int> neighbors;


    // float rcs_m = std::sqrt(std::pow(10, point.intensity / 10));
    // float max_rcs_m = std::sqrt(std::pow(10, max_int / 10));
    // float min_rcs_m = std::sqrt(std::pow(10, min_int / 10));
    // float norm_rcs_m =  (rcs_m - min_rcs_m) / (max_rcs_m - min_rcs_m);  
    // currently we are not using the dynamic epsilon logic.
    // But this would be a possible optimisation approach
    float dynamic_eps = epsilon;  

    // if (dynamic_eps < 0.01)
    // {
    //     dynamic_eps = 0.01;
    // }

    // if (dynamic_eps < epsilon)
    // {
    //     dynamic_eps = epsilon;
    // }

    for (int i = 0; i < points.size(); ++i) 
    {  
        if (distance(point, points[i]) <= dynamic_eps) 
        {            
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

// function to find maximum intensity in the points set
float DBSCAN_cluster::find_intensity_bounds(std::vector<datastructures::Point>& points, float& max_int, float& min_int)
{
    auto max_intensity_point = std::max_element(points.begin(), points.end(),[](const datastructures::Point& a, const datastructures::Point& b) 
    {
        return a.rcs < b.rcs; // Comparison based on intensity
    });

    max_int = static_cast<float>(max_intensity_point->rcs);

    auto min_intensity_point = std::min_element(points.begin(), points.end(),[](const datastructures::Point& a, const datastructures::Point& b) 
    {
            return a.rcs < b.rcs; // Comparison based on intensity
    });

    min_int = static_cast<float>(min_intensity_point->rcs);

}

// Expand the cluster recursively based on neighbors
bool DBSCAN_cluster::expandCluster(std::vector<datastructures::Point>& points, int point_idx, int cluster_id)
{   

    float max_int, min_int;
    find_intensity_bounds(points, max_int, min_int);
    auto neighbors = regionQuery(points, points[point_idx], max_int, min_int);

    // If the number of neighbors is below the minimum number of points, mark as noise
    if (neighbors.size() < min_points) 
    {
        points[point_idx].cluster_id = -2; // Noise
        return false;
    }

    // Assign the current point to the cluster
    points[point_idx].cluster_id = cluster_id;

    // Process all neighbors
    std::vector<int> seed_set = neighbors;

    // Loop over the seed set
    for (size_t i = 0; i < seed_set.size(); ++i) {
        int neighbor_idx = seed_set[i];

        // If this neighbor has not been visited, mark it visited and check its neighbors
        if (!points[neighbor_idx].visited) 
        {
            points[neighbor_idx].visited = true;

            auto new_neighbors = regionQuery(points, points[neighbor_idx], max_int, min_int);

            // If the neighbor has a large enough neighborhood, merge them
            if (new_neighbors.size() >= min_points) 
            {
                seed_set.insert(seed_set.end(), new_neighbors.begin(), new_neighbors.end());
            }
        }

        // If the neighbor hasn't been assigned to a cluster, assign it
        if (points[neighbor_idx].cluster_id == -1) 
        {
            points[neighbor_idx].cluster_id = cluster_id;
        }
    }


    return true;
}

// Main DBSCAN algorithm function
void DBSCAN_cluster::fit(std::vector<datastructures::Point>& points)
{
    int cluster_id = 0;

    // Iterate over each point in the dataset
    for (int i = 0; i < points.size(); ++i) {
        // If the point has already been visited, skip it
        if (points[i].visited) 
        {
            continue;
        }

        points[i].visited = true;

        // Try to expand the cluster from the current point
        if (expandCluster(points, i, cluster_id)) 
        {
            cluster_id++; // Move to the next cluster
        }
    }
}
} // namespace 

#endif // DBSCAN_HPP
