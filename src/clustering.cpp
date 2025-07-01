#include <chrono>
#include <cmath>
#include <Eigen/Geometry>
#include <cstring>
#include <random>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "../include/master_thesis/datastructure.hpp"
#include "../include/master_thesis/dbscan.hpp"
#include "../include/master_thesis/helperfunctions.hpp"


class   ClustererNode :   public  rclcpp::Node
{
    public:
        ClustererNode() : Node("clustering_node")
        {

            this->declare_parameter<double>("dbscan.epsilon", 1.0);
            this->declare_parameter<int>("dbscan.min_pts", 2);
            this->declare_parameter<int>("dbscan.msg_count", 1);
            this->declare_parameter<double>("radar.angle_max", 1.0471975512);
            this->declare_parameter<double>("radar.angle_min", 0.0);
            this->declare_parameter<double>("radar.angle_var_max", 0.034906585);
            this->declare_parameter<double>("radar.angle_var_min", 0.0087266463);
            this->declare_parameter<double>("radar.range_var", 0.15);
            this->declare_parameter<double>("visualization.scale_factor", 1.0);
            this->declare_parameter<double>("visualization.min_diameter", 0.1);
            this->declare_parameter<int>("visualization.history", 20);
            this->declare_parameter<double>("visualization.alpha", 0.7);
            this->declare_parameter<double>("visualization.history_alpha", 0.1);
            this->declare_parameter<double>("map.cell_size", 0.25);
            this->declare_parameter<double>("map.map_size", 20);
            this->declare_parameter<double>("gaussian.velocity_threshold", 2.0);
            

            eps             =   this->get_parameter("dbscan.epsilon").as_double();
            min_pts         =   this->get_parameter("dbscan.min_pts").as_int();
            count           =   this->get_parameter("dbscan.msg_count").as_int();
            angle_max       =   this->get_parameter("radar.angle_max").as_double();
            angle_min       =   this->get_parameter("radar.angle_min").as_double();
            angle_var_max   =   this->get_parameter("radar.angle_var_max").as_double();
            angle_var_min   =   this->get_parameter("radar.angle_var_min").as_double();
            range_var       =   this->get_parameter("radar.range_var").as_double();
            scale_factor    =   this->get_parameter("visualization.scale_factor").as_double();
            min_diameter    =   this->get_parameter("visualization.min_diameter").as_double();
            history         =   this->get_parameter("visualization.history").as_int();
            alpha           =   this->get_parameter("visualization.alpha").as_double();
            history_alpha   =   this->get_parameter("visualization.history_alpha").as_double();
            cell_size       =   this->get_parameter("map.cell_size").as_double();
            map_size        =   this->get_parameter("map.map_size").as_double();
            vel_thresh      =   this->get_parameter("gaussian.velocity_threshold").as_double();


            num_cells       =   static_cast<int>(map_size/cell_size);
            grid_rows       =   num_cells;
            grid_cols       =   num_cells;
            occ_grid        =   std::vector<std::vector<float>>(grid_rows, std::vector<float>(grid_cols, 0.5f));


            pcl_sub         =   this -> create_subscription<sensor_msgs::msg::PointCloud2>
                                ("/radarpcl_agg", rclcpp::SystemDefaultsQoS(), std::bind(&ClustererNode::pointcloud_callback, this, std::placeholders::_1));
            
            odom_sub        =   this -> create_subscription<nav_msgs::msg::Odometry>
                                ("/odom", rclcpp::SystemDefaultsQoS(), std::bind(&ClustererNode::odometry_callback, this, std::placeholders::_1));
            
            dbscan          =   std::make_shared<clustering::DBSCAN_cluster>(eps, min_pts);  
            
            ellipse_pub     =   this -> create_publisher<visualization_msgs::msg::MarkerArray>("ellipse", rclcpp::SystemDefaultsQoS()); 
            
            pointcloud_pub  =   this -> create_publisher<sensor_msgs::msg::PointCloud2>("clustered_cloud", rclcpp::SystemDefaultsQoS()); 

            odom_pub        =   this -> create_publisher<nav_msgs::msg::Odometry>("odom_corrected", rclcpp::SystemDefaultsQoS());

            dynamic_tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
            static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        }
    
    private:

        
        int min_pts;
        int count;
        int grid_rows;
        int grid_cols;
        int num_cells;
        int history;
        double eps;
        double angle_max;
        double angle_min;
        double angle_var_max;
        double angle_var_min;
        double range_var;
        double scale_factor;
        double min_diameter;
        double alpha;
        double history_alpha;
        double cell_size;
        double map_size;
        double vel_thresh;

        
        std::vector<datastructures::Point> radar_data;
        Eigen::Vector3f origin;
        int msg_count   =   0;
        bool origin_set =   false;
        datastructures::GaussianMixture unique_gaussians;
        datastructures::MarkerHistory marker_queue;
        std::vector<std::vector<float>> occ_grid;
        nav_msgs::msg::Odometry odom_msg;


        void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
        {
            
            // geometry_msgs::msg::TransformStamped t;
            if(!origin_set)
            {
                origin      <<  msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
                origin_set  =   true;

                geometry_msgs::msg::TransformStamped t;
                t.header.stamp = this->get_clock()->now();
                t.header.frame_id = "map";
                t.child_frame_id = "map_corr";
                t.transform.translation.x = origin(0);
                t.transform.translation.y = origin(1);
                t.transform.translation.z = origin(2);
                t.transform.rotation.w = 1.0; // Identity rotation
                static_tf_broadcaster_->sendTransform(t);
            }

            if(origin_set)
            {
                
                odom_msg.child_frame_id         =   "base_link";
                odom_msg.header.frame_id        =   "map_corr";
                odom_msg.header.stamp           =   msg->header.stamp;
                odom_msg.pose.covariance        =   msg->pose.covariance; 
                odom_msg.pose.pose.orientation  =   msg->pose.pose.orientation;
                odom_msg.pose.pose.position.x   =   msg->pose.pose.position.x - origin(0);
                odom_msg.pose.pose.position.y   =   msg->pose.pose.position.y - origin(1);
                odom_msg.pose.pose.position.z   =   msg->pose.pose.position.z - origin(2);
                odom_msg.twist                  =   msg->twist;

                geometry_msgs::msg::TransformStamped t;

                // Read the transform from the odom_msg
                t.header.stamp = odom_msg.header.stamp;
                t.header.frame_id = "map_corr"; // "map_corr"
                t.child_frame_id = "base_link";

                t.transform.translation.x = odom_msg.pose.pose.position.x;
                t.transform.translation.y = odom_msg.pose.pose.position.y;
                t.transform.translation.z = odom_msg.pose.pose.position.z;
                t.transform.rotation = odom_msg.pose.pose.orientation;

                // Send the transform
                odom_pub->publish(odom_msg);
                dynamic_tf_broadcaster_->sendTransform(t);

            }
        }

        void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {

            if(!origin_set)
            {
                return;
            }

            msg_count += 1;
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
            sensor_msgs::PointCloud2ConstIterator<float> iter_range(*msg, "range");
            sensor_msgs::PointCloud2ConstIterator<float> iter_azimuth(*msg, "azimuth");
            sensor_msgs::PointCloud2ConstIterator<float> iter_velocity(*msg, "vr");
            sensor_msgs::PointCloud2ConstIterator<float> iter_velocity_compensated(*msg, "vr_c");
            sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "rcs");
            sensor_msgs::PointCloud2ConstIterator<float> iter_label(*msg, "label");

            for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_range, ++iter_azimuth, ++iter_velocity, ++iter_velocity_compensated, ++iter_intensity, ++iter_label) {

                double   x           =   static_cast<double>(*iter_x);
                double   y           =   static_cast<double>(*iter_y);
                double   z           =   static_cast<double>(*iter_z);
                double   range       =   static_cast<double>(*iter_range);
                double   azimuth     =   static_cast<double>(*iter_azimuth);
                double   velocity    =   static_cast<double>(*iter_velocity);
                double   velocity_c  =   static_cast<double>(*iter_velocity_compensated);
                double   intensity   =   static_cast<double>(*iter_intensity);
                uint8_t label        =   *iter_label;
                
                // transforming cordinates relative to first odometry position
                x   -=  origin(0);
                y   -=  origin(1);
                z   -=  origin(2);

                // rcs values are in dBsm, so it can be negative, this will cause issues in weight calculation
                // converting from logarithmic to linear scale
                double rcs_linear   = std::max(1e-6, std::pow(10.0, intensity / 10.0)); 
                datastructures::Point radar_point;
                radar_point.x           =   x;
                radar_point.y           =   y;
                radar_point.z           =   z;
                radar_point.azimuth     =   azimuth;
                radar_point.range       =   range;
                radar_point.vel         =   velocity_c;
                radar_point.rcs         =   rcs_linear;   
                
                radar_data.push_back(radar_point);

            }

            if (msg_count > count)
            {
                // perform dbscan filtering 
                auto start      =   std::chrono::high_resolution_clock::now();
                dbscan          ->  fit(radar_data);
                auto end        =   std::chrono::high_resolution_clock::now();
                auto duration   = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // std::cout << "\ndata after dbscan (range, azimuth): (" << radar_data[0].range << ", " << radar_data[0].azimuth  << ")" << std::flush;
                // std::cout << "dbscan time: " << duration.count() << " microseconds\n";

                // Arranging data into defined datastructure
                datastructures::ClusterMap cluster_data;
                start           = std::chrono::high_resolution_clock::now();

                cluster_data    = helperfunctions::create_dataframe(radar_data);
                // std::cout << "\n data after dataframe creation (range, azimuth): (" << cluster_data[0].points[0].range << ", " << cluster_data[0].points[0].azimuth << ")" << std::flush;
                end             = std::chrono::high_resolution_clock::now();
                duration        = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // std::cout << "dataframe creation time: " << duration.count() << " microseconds\n";
                
                // creating the gaussian mixture from the clusters
                datastructures::GaussianMixture gaussian_mixture;
                gaussian_mixture    =   helperfunctions::create_gaussian_mixture(cluster_data, angle_max, angle_min, angle_var_max, angle_var_min, range_var, vel_thresh);
                rclcpp::Time stamp  =   msg->header.stamp;
                int total_points    =   radar_data.size();

                // identifying unique clusters

                if(unique_gaussians.empty())
                {
                    unique_gaussians    =   gaussian_mixture;
                }
                else
                {
                    // helperfunctions::update_unique_gaussians(unique_gaussians, gaussian_mixture);
                }
                publish_ellipsoid(gaussian_mixture, stamp);
                publish_clustered_pointcloud(cluster_data, stamp);

                radar_data.clear();
                msg_count = 0;
            }
        }

        void publish_ellipsoid(const datastructures::GaussianMixture& gaussian_mixture, const rclcpp::Time& stamp)
        {
            visualization_msgs::msg::MarkerArray marker_array_msg;
            std_msgs::msg::Header header;
            header.stamp = stamp; 
            header.frame_id = "map_corr";
            float scale_factor_ = static_cast<float>(scale_factor);
            float min_diameter_ = static_cast<float>(min_diameter);
            float alpha_        = static_cast<float>(alpha);
            float history_alpha_= static_cast<float>(history_alpha);


            helperfunctions::create_gaussian_ellipsoid_markers(gaussian_mixture, odom_msg, marker_array_msg, marker_queue, header, scale_factor_, min_diameter_, alpha_, history_alpha_, history);
            ellipse_pub->publish(marker_array_msg);  
        }


        void publish_clustered_pointcloud(const datastructures::ClusterMap& clusters, const rclcpp::Time& stamp)
        {
            int num_points = 0;
            for (auto& entry    :   clusters)
            {
                num_points += static_cast<int>(entry.second.points.size());
            }

            sensor_msgs::msg::PointCloud2 cloud_msg;
            cloud_msg.header.stamp = stamp;
            cloud_msg.header.frame_id = "map_corr";

            

            // Define additional fields for RGB
            int rgb_offset = sizeof(float) * 3; // Offset after xyz and cluster_id

            // Creating fileds
            cloud_msg.height = 1;
            cloud_msg.width = num_points;
            cloud_msg.is_bigendian = false;
            cloud_msg.is_dense = true;
            cloud_msg.point_step = sizeof(float) * 3 + sizeof(uint32_t);
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

            // Defining the fields (x, y, z, cluster_id)
            sensor_msgs::msg::PointField field;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;

            field.name = "x";
            field.offset = 0;
            cloud_msg.fields.push_back(field);

            field.name = "y";
            field.offset = 4;
            cloud_msg.fields.push_back(field);

            field.name = "z";
            field.offset = 8;
            cloud_msg.fields.push_back(field);

            // RGB field definition
            sensor_msgs::msg::PointField rgb_field;
            rgb_field.datatype = sensor_msgs::msg::PointField::UINT32; // 32-bit unsigned integer for RGB
            rgb_field.count = 1; // Now only 1 count, as it's a single 32-bit value
            rgb_field.name = "rgb";
            rgb_field.offset = rgb_offset;
            cloud_msg.fields.push_back(rgb_field);

            // Allocate data
            cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

            size_t current_byte_offset = 0;
            for(const auto& entry : clusters)
            {
                int cluster_id = entry.first;
                datastructures::ClusterData data = entry.second;
                std::vector<datastructures::Point> point_data = data.points;

                std_msgs::msg::ColorRGBA color_rgba;
                color_rgba          =   helperfunctions::generateColorFromGaussianID(cluster_id, 1.0);
                uint8_t r_u8        =   static_cast<uint8_t>(color_rgba.r * 255.0f);
                uint8_t g_u8        =   static_cast<uint8_t>(color_rgba.g * 255.0f);
                uint8_t b_u8        =   static_cast<uint8_t>(color_rgba.b * 255.0f);
                uint8_t a_u8        =   static_cast<uint8_t>(color_rgba.a * 255.0f);

                uint32_t rgb_color =    (static_cast<uint32_t>(a_u8) << 24) |
                                        (static_cast<uint32_t>(r_u8) << 16) |
                                        (static_cast<uint32_t>(g_u8) << 8)  |
                                        (static_cast<uint32_t>(b_u8));

                for(size_t j = 0; j < point_data.size(); ++j)
                {                       
                    uint8_t *data_ptr = &cloud_msg.data[current_byte_offset];
                    float px = static_cast<float>(point_data[j].x);
                    float py = static_cast<float>(point_data[j].y);
                    float pz = static_cast<float>(point_data[j].z);                
                    memcpy(data_ptr + 0, &px, sizeof(float));
                    memcpy(data_ptr + 4, &py, sizeof(float));
                    memcpy(data_ptr + 8, &pz, sizeof(float));
                    memcpy(data_ptr + rgb_offset, &rgb_color, sizeof(uint32_t));
                    current_byte_offset += cloud_msg.point_step;                    
                } 


            }
            pointcloud_pub->publish(cloud_msg);

            
        }


        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
        std::shared_ptr<clustering::DBSCAN_cluster> dbscan;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ellipse_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
        std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_tf_broadcaster_;
        std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;


};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);    
    rclcpp::spin(std::make_shared<ClustererNode>());
    rclcpp::shutdown();
    return 0;
}