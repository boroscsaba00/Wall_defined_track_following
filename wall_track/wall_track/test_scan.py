import rclpy
from rclpy.node import Node
import copy
import math

from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pcl2
from visualization_msgs.msg import MarkerArray, Marker

class ScanHandler(Node):
    def __init__(self):
        super().__init__('test_viz')

        self.pub_cloud = self.create_publisher(PointCloud2, "/cloud", 1)

        self.pub_marker = self.create_publisher(MarkerArray, "/viz", 1)

        self.closest_point_pub = self.create_publisher(PoseStamped, "/closest_point", 1)
        
        self.sub = self.create_subscription(LaserScan, "/scan", self.callback_scan, 1)

    def callback_scan(self, scan_in:LaserScan):
        
        newPoint = []
        for i in range(len(scan_in.ranges)):
            
            if ((scan_in.range_min < scan_in.ranges[i]) and (scan_in.ranges[i] < scan_in.range_max)):
                angle = scan_in.angle_min + scan_in.angle_increment * i
                x = scan_in.ranges[i] * math.cos(angle)
                y = scan_in.ranges[i] * math.sin(angle)
                z = 0.0
                newPoint.append([x, y, z])
        
        ## MarkerArray
        marker_array_ = MarkerArray()
        
        marker = Marker()
        marker.header = scan_in.header
        marker.ns = "scan"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.a = 1.0
        marker.color.b = 1.0

        for point in newPoint:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]

            marker.points.append(p)

        marker.lifetime = rclpy.duration.Duration(seconds=0.15).to_msg()

        marker_array_.markers.append(marker)

        self.pub_marker.publish(marker_array_)

        ## PointCloud

        localCloud = pcl2.create_cloud_xyz32(scan_in.header, newPoint)
        self.pub_cloud.publish(localCloud)

        ## Closest point
        closest_dist = 1000
        closest_dist_angle = 0

        for i in range(len(scan_in.ranges)):
            dist = scan_in.ranges[i] 
            if dist < closest_dist:
                closest_dist = dist
                closest_dist_angle = scan_in.angle_min + i * scan_in.angle_increment

        closest_point = PoseStamped()
        closest_point.header = scan_in.header
        closest_point.pose.position.x = closest_dist * math.cos(closest_dist_angle)
        closest_point.pose.position.y = closest_dist * math.sin(closest_dist_angle)
        closest_point.pose.position.z = 0.0
        closest_point.pose.orientation.w = 1.0
        self.closest_point_pub.publish(closest_point)




def main(args=None):
    rclpy.init(args=args)
    scan_handler_ = ScanHandler()
    rclpy.spin(scan_handler_)
    scan_handler_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()