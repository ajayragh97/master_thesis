import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import json


class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')

        self.dataset_path = "/home/ajay/work/Fraunhofer/dataset/RadarScenes/data"
        self.seq = 145
        self.meta_file = "{}/sequence_{}/scenes.json".format(self.dataset_path, self.seq)

        with open(self.meta_file, 'r') as file:
            self.meta_data = json.load(file)

        self.pcl_sub = self.create_subscription(PointCloud2, "clustered_cloud", self.pcl_callback, 10)
        self.image_pub = self.create_publisher(Image, 'image', 10)
        self.bridge = CvBridge()
        self.frame_id = 'map'

    def pcl_callback(self, msg):
        stamp   =   msg.header.stamp
        key     =   int((stamp.sec * 1_000_000_000 + stamp.nanosec) / 1000)
        
        self.image_name = self.meta_data['scenes'][f'{key}']['image_name']
        image_path = f"{self.dataset_path}/sequence_{self.seq}/camera/{self.image_name}"

        self.cv_image = cv2.imread(image_path)
        self.publish_image()

    def publish_image(self):
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8")
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            ros_image_msg.header.frame_id = self.frame_id

            self.image_pub.publish(ros_image_msg)
            print(f"publishing {self.image_name}")
        
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")



def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()

    try:
        while rclpy.ok():
            rclpy.spin_once(image_publisher, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass


    # Shutdown
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
