#ifndef DATASTRUCTURE_HPP
#define DATASTRUCTURE_HPP

#include <map>
#include <vector>
#include <Eigen/Dense>



namespace datastructures
{
    struct Point
    {
        int             cluster_id = -1;
        double          x;
        double          y;
        double          z;
        double          range;
        double          azimuth;
        double          rcs;
        double          vel;
        Eigen::Matrix3d cov;
        Eigen::Matrix2d cov_2d;
        bool visited = false;
    };

    struct ClusterData
    {
        std::vector<Point> points;
        Eigen::Matrix3d     cov;
        Eigen::Matrix2d     cov_2d;
        Eigen::RowVectorXd  center;
        double               avg_vel;
    };

    using   ClusterMap =   std::map<int, ClusterData>;

    struct Gaussian
    {
        Eigen::Vector3d centroid;
        Eigen::Matrix3d cov;
        Eigen::Matrix2d cov_2d;
        Eigen::Matrix4d pose;
        Eigen::Vector3d eigenvalues;
        int id;
        double intensity;
        double noise;
        double weight;
        double velocity;
        bool dynamic = false;
    };

    using GaussianMixture = std::map<int, Gaussian>;

    struct Ellipse2D
    {
        double axis_1;
        double axis_2;
        double orientation;
    };
}

#endif