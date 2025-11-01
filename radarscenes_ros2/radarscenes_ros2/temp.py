import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, TransformStamped
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
import numpy as np
import h5py
import json
import os
import time



class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher')
        self.publisher_cc = self.create_publisher(PointCloud2, 'radarpcl_cc', 10)
        self.publisher_gc = self.create_publisher(PointCloud2, 'radarpcl_gc', 10)
        self.publisher_odom = self.create_publisher(Odometry, 'odom', 10)
        self.publisher_tf = TransformBroadcaster(self)
                
        self.dataset_path = "/home/ajay/work/Fraunhofer/dataset/RadarScenes/data"
        self.seq = 1
        self.data_file = "{}/sequence_{}/radar_data.h5".format(self.dataset_path, self.seq)
        self.meta_file = "{}/sequence_{}/scenes.json".format(self.dataset_path, self.seq)
        self.sensor_file = "{}/sensors.json".format(self.dataset_path)
        
        with h5py.File(self.data_file, 'r') as file:

            self.radar_data = file['radar_data'][:]
            self.odom_data = file['odometry'][:]

        with open(self.meta_file, 'r') as file:
            self.meta_data = json.load(file)
        
        with open(self.sensor_file, 'r') as file:
            self.sensor_data = json.load(file)

        self.current_odom_idx = 0
        self.current_pcl_idx = 0

    def create_pcl_msg(self):

    

        scenes = self.meta_data['scenes']
        timestamps = scenes.keys()

        for idx,stamp in enumerate(timestamps):
            sensor_id = scenes[stamp]['sensor_id']
            start_idx = scenes[stamp]['radar_indices'][0]
            end_idx = scenes[stamp]['radar_indices'][1]
            
            cc_data = []
            gc_data = [] 
            for i in range(start_idx, end_idx):
                
                rcs = self.radar_data[i][4]
                vr = self.radar_data[i][5]
                vr_c = self.radar_data[i][6]
                x_cc = self.radar_data[i][7]
                y_cc = self.radar_data[i][8]
                z_cc = 0.0
                x_gc = self.radar_data[i][9]
                y_gc = self.radar_data[i][10]
                z_gc = 0.0
                label = self.radar_data[i][13]

                cc = [x_cc, y_cc, z_cc, vr, vr_c, rcs, label]
                gc = [x_gc, y_gc, z_gc,  vr, vr_c, rcs, label]

                cc_data.append(cc)
                gc_data.append(gc)
            
            cc_data = np.asarray(cc_data)
            gc_data = np.asarray(gc_data)

            # publish point clouds
            time_delay = int(list(timestamps)[idx+1]) - int(stamp)  # time delay in micorseconds
            # print("pcl delay : {}".format(time_delay))
            time.sleep(time_delay / 1_000_000)

            self.publish_pointcloud(cc_data, "radar_{}".format(sensor_id), stamp)
            self.publish_pointcloud(gc_data, "map", stamp)
        
    def publish_pointcloud(self, data, frame_id, timestamp):
        # Define cloud properties
        cloud_msg = PointCloud2()
        timestamp = int(timestamp)
        timestamp_msg = Time()
        seconds = timestamp // 1_000_000  # Integer division for seconds
        nanoseconds = (timestamp % 1_000_000) * 1_000  # Remaining microseconds converted to nanoseconds
        timestamp_msg.sec = int(seconds)
        timestamp_msg.nanosec = int(nanoseconds)
        cloud_msg.header.stamp = timestamp_msg
        cloud_msg.header.frame_id = frame_id  # Set the frame ID

        # Define point cloud fields (x, y, vr, vr_c, rcs, label)
        cloud_msg.height = 1
        cloud_msg.width = data.shape[0]
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = False
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr_c', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='rcs', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=24, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg.point_step = 28  # Each point has 7 fields x, y, z, vr, vr_c, rcs, label (4 bytes each)
        cloud_msg.row_step = cloud_msg.point_step * data.shape[0]

        cloud_msg.data = data.tobytes()  # Convert to binary format

        # Publish the PointCloud2 message
        if(frame_id == "map"):
            self.publisher_gc.publish(cloud_msg)
        else:
            self.publisher_cc.publish(cloud_msg)

    def publish_odom_msg(self):

        odom_msg = Odometry()
        tf_msg = TransformStamped()
        
        for i in range(self.odom_data.shape[0]):

            timestamp = self.odom_data[i][0]
            timestamp_msg = Time()
            seconds = timestamp // 1_000_000  # Integer division for seconds
            nanoseconds = (timestamp % 1_000_000) * 1_000  # Remaining microseconds converted to nanoseconds
            timestamp_msg.sec = int(seconds)
            timestamp_msg.nanosec = int(nanoseconds)
            odom_msg.header.stamp = timestamp_msg
            tf_msg.header.stamp = timestamp_msg

            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'
            tf_msg.header.frame_id = 'map'
            tf_msg.child_frame_id = 'base_link'

            odom_msg.pose.pose.position.x = float(self.odom_data[i][1])
            odom_msg.pose.pose.position.y = float(self.odom_data[i][2])
            odom_msg.pose.pose.position.z = 0.0

            tf_msg.transform.translation.x = float(self.odom_data[i][1])
            tf_msg.transform.translation.y = float(self.odom_data[i][2])
            tf_msg.transform.translation.z = 0.0

            yaw_gc = float(self.odom_data[i][3])
            q = quaternion_from_euler(0, 0, yaw_gc)
            
            odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            tf_msg.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


            if i < self.odom_data.shape[0] - 1:
                time_delay_ = self.odom_data[i + 1][0] - timestamp
            else:
                time_delay_ = 0
            print("odom delay: {}".format(time_delay_))
            time.sleep(time_delay_ / 1_000_000)
            self.publisher_odom.publish(odom_msg)
            self.publisher_tf.sendTransform(tf_msg)


        

def main(args=None):
    rclpy.init(args=args)
    pointcloud_publisher = PointCloudPublisher()

    try:
        while rclpy.ok():
            pointcloud_publisher.create_pcl_msg()
            pointcloud_publisher.publish_odom_msg()
            rclpy.spin_once(pointcloud_publisher, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass


    # Shutdown
    pointcloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
