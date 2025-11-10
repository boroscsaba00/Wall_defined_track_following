import rclpy
from rclpy.node import Node
import math
import numpy as np

# ROS 2 üzenettípusok
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2 as pcl2
from std_msgs.msg import Header

# TF2 modulok
import tf2_ros
from tf_transformations import quaternion_from_euler
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud 


class TurtleBot3WallFollower(Node):
    
    def __init__(self):
        super().__init__('wall_follower_node')
        
        self.declare_parameter('target_frame', 'odom')
        self.target_frame = self.get_parameter('target_frame').value 

        self.L = 0.287

        self.Kp = 0.8
        self.Ki = 0.05
        self.Kd = 0.1
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.target_distance = 0.35 
        self.LINEAR_SPEED = 0.20

        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.cmd = Twist()

        self.dt = 0.05 # 20 Hz

        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 1)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 1)
        self.pub_odom_left = self.create_publisher(Odometry, '/odom_wheel_left', 1) 
        self.pub_odom_right = self.create_publisher(Odometry, '/odom_wheel_right', 1) 
        self.pub_cloud = self.create_publisher(PointCloud2, "/cloud_map", 1)

        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.callback_scan, 1)

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.scan_data = LaserScan()

        self.cmd.linear.x = 0.05
        
        self.pub_cmd.publish(self.cmd)


    def callback_scan(self, msg: LaserScan):
        """Beérkező LaserScan üzenet elmentése."""
        self.scan_data = msg


    def calculate_control(self):
        """Kozeppont kereses ket fal kozott"""
        scan = self.scan_data
        
        if not scan.ranges:
            self.get_logger().warn("No scanned data")
            return 0.0

        ranges = np.array(scan.ranges)
        
        ranges[np.isinf(ranges) | np.isnan(ranges)] = scan.range_max

        RIGHT_ANGLE_MAX = -60 * math.pi/180
        RIGHT_ANGLE_MIN = -90 * math.pi/180
        
        LEFT_ANGLE_MIN = 60 * math.pi/180
        LEFT_ANGLE_MAX = 90 * math.pi/180

        
        def angle_to_index(angle_rad):
            idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
            
            return max(0, min(len(ranges) - 1, idx))

        
        idx_R_min = angle_to_index(RIGHT_ANGLE_MIN)
        idx_R_max = angle_to_index(RIGHT_ANGLE_MAX)
        
        idx_L_min = angle_to_index(LEFT_ANGLE_MIN)
        idx_L_max = angle_to_index(LEFT_ANGLE_MAX)

        if idx_R_min < idx_R_max:
            dist_right = np.min(ranges[idx_R_min:idx_R_max])
        else: 
             dist_right = np.min(ranges[idx_R_max:idx_R_min])
             
        if idx_L_min < idx_L_max:
             dist_left = np.min(ranges[idx_L_min:idx_L_max])
        else:
            dist_left = np.min(ranges[idx_L_max:idx_L_min])

        
        MAX_DIST = 10.0 # Define a high value for open space
        
        # Replace the capping logic with this:
        if dist_right >= scan.range_max * 0.9: # If reading max range (open space)
            dist_right = MAX_DIST
        
        if dist_left >= scan.range_max * 0.9: # If reading max range (open space)
            dist_left = MAX_DIST

        
        error = dist_left - dist_right 
        
        
        self.error_integral += error * self.dt
        error_derivative = (error - self.previous_error) / self.dt
        
        angular_output = self.Kp * error + self.Ki * self.error_integral + self.Kd * error_derivative
        self.previous_error = error
        
        idx_front = angle_to_index(0) 
        dist_front = ranges[idx_front] if ranges[idx_front] < scan.range_max else 10.0

        LINEAR_SPEED = self.LINEAR_SPEED

        # Define a safety zone
        SAFETY_DIST = 0.6
        STOP_DIST = 0.3
        
        if dist_front < STOP_DIST:
            linear_output = 0.0
            angular_output = np.sign(angular_output) * 1.0 if abs(angular_output) > 0.1 else 1.0 

        elif dist_front < SAFETY_DIST:
            slowdown_ratio = (dist_front - STOP_DIST) / (SAFETY_DIST - STOP_DIST)
            
            linear_output = self.LINEAR_SPEED * slowdown_ratio
            
        else:
            linear_output = self.LINEAR_SPEED
            
        linear_output = np.clip(linear_output, 0.0, self.LINEAR_SPEED)

        self.cmd.linear.x = linear_output
        self.cmd.angular.z = angular_output
        
        angular_output = np.clip(angular_output, -1.5, 1.5)

        self.cmd.linear.x = linear_output
        self.cmd.angular.z = angular_output
        
        self.pub_cmd.publish(self.cmd)
        
        return angular_output
      


    def publish_odom(self, frame_id: str, child_frame_id: str, x: float, y: float, yaw: float, publisher):
        """Odometria üzenet és TF transzformáció közzététele."""

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = frame_id
        odom.child_frame_id = child_frame_id
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y

    
        q = quaternion_from_euler(0, 0, yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        publisher.publish(odom)


    def timer_callback(self):
        """Fő ciklus: Vezérlés, Szimuláció, Odometria, Pontfelhő."""
        
        self.calculate_control()

        self.x += self.cmd.linear.x * self.dt * math.cos(self.yaw)
        self.y += self.cmd.linear.x * self.dt * math.sin(self.yaw)
        self.yaw += self.cmd.angular.z * self.dt

        self.publish_odom(self.target_frame, "base_link", self.x, self.y, self.yaw, self.pub_odom)
        
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = self.get_clock().now().to_msg()
        tf_stamped.header.frame_id = self.target_frame
        tf_stamped.child_frame_id = "base_link"
        tf_stamped.transform.translation.x = self.x
        tf_stamped.transform.translation.y = self.y
        tf_stamped.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, self.yaw)
        tf_stamped.transform.rotation.x = q[0]
        tf_stamped.transform.rotation.y = q[1]
        tf_stamped.transform.rotation.z = q[2]
        tf_stamped.transform.rotation.w = q[3]
        self.broadcaster.sendTransform(tf_stamped)

        self.publish_wheel_odom("wheel_left_link", self.pub_odom_left)
        self.publish_wheel_odom("wheel_right_link", self.pub_odom_right)
        
        
        self.publish_point_cloud()


    def publish_wheel_odom(self, wheel_frame: str, publisher):
        """A kerék TF transzformációjának lekérése és Odometria üzenetként való közzététele."""
        
        now = rclpy.time.Time()
        
        if self.tfBuffer.can_transform(self.target_frame, wheel_frame, time=now, timeout=rclpy.duration.Duration(seconds=0.1)):
            trans = self.tfBuffer.lookup_transform(self.target_frame, wheel_frame, now)

            odom = Odometry()
            odom.header.stamp = trans.header.stamp
            odom.header.frame_id = self.target_frame
            odom.child_frame_id = trans.child_frame_id
            odom.pose.pose.position.x = trans.transform.translation.x
            odom.pose.pose.position.y = trans.transform.translation.y
            odom.pose.pose.orientation = trans.transform.rotation

            publisher.publish(odom)
         #else
             #self.get_logger().warn(f"TF transzformáció ({self.target_frame} -> {wheel_frame}) nem érhető el.")


    def publish_point_cloud(self):
        """A LaserScan adatok átalakítása PointCloud2 üzenetté és közzététele a térkép (odom) frame-ben."""
        
        scan = self.scan_data
        if not scan.ranges:
            return

        
        points = []
        for i in range(len(scan.ranges)):
            if scan.range_min < scan.ranges[i] < scan.range_max:
                angle = scan.angle_min + scan.angle_increment * i
                x = scan.ranges[i] * math.cos(angle)
                y = scan.ranges[i] * math.sin(angle)
                z = 0.0
                points.append([x, y, z])

        
        cloud_header = Header()
        cloud_header.frame_id = scan.header.frame_id 
        cloud_header.stamp = self.get_clock().now().to_msg()
        localCloud = pcl2.create_cloud_xyz32(cloud_header, points)

        now = rclpy.time.Time()
        if self.tfBuffer.can_transform(self.target_frame, scan.header.frame_id, time=now, timeout=rclpy.duration.Duration(seconds=0.1)):
            trans = self.tfBuffer.lookup_transform(self.target_frame, scan.header.frame_id, now)
            
            
            mapCloud = do_transform_cloud(localCloud, trans)
            
            
            self.pub_cloud.publish(mapCloud)
        # else:
            # self.get_logger().warn(f"TF transzformáció ({self.target_frame} -> {scan.header.frame_id}) nem érhető el.")


def main(args=None):
    rclpy.init(args=args)
    tb3_follower = TurtleBot3WallFollower()
    rclpy.spin(tb3_follower)
    tb3_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()