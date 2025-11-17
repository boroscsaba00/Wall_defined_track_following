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
        self.Kp = 1.0
        self.Ki = 0.001
        self.Kd = 0.5     
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.TARGET_LINEAR_SPEED = 0.2

        self.SECTOR_LEFT_MIN = 10
        self.SECTOR_LEFT_MAX = 90
        self.SECTOR_RIGHT_MIN = 270
        self.SECTOR_RIGHT_MAX = 350
        self.SECTOR_FRONT_MIN = 330
        self.SECTOR_FRONT_MAX = 30

        
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


    def callback_scan(self, msg: LaserScan):
        """Beérkező LaserScan üzenet elmentése."""
        self.scan_data = msg

    #filtering out irrelevant data
    def get_sector_min_distance(self, scan_data: LaserScan, min_deg: int, max_deg: int):
        
        ranges = np.array(scan_data.ranges)
        angle_min_rad = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        
        N = len(ranges)
        
        start_index = int((np.deg2rad(min_deg) - angle_min_rad) / angle_increment)
        end_index = int((np.deg2rad(max_deg) - angle_min_rad) / angle_increment)
        data_slice = []
        if min_deg > max_deg: 
            data_slice = np.concatenate((ranges[start_index:], ranges[:end_index]))
        else:
            start_index = np.clip(start_index, 0, N)
            end_index = np.clip(end_index, 0, N)
            data_slice = ranges[start_index:end_index]
            
        valid_distances = data_slice[(data_slice > scan_data.range_min) & (data_slice < scan_data.range_max)]
        
        if valid_distances.size > 0:
            return np.min(valid_distances)
        else:
            return scan_data.range_max * 0.9 

            
    def calculate_control(self):
        scan = self.scan_data
        
        if not scan.ranges:
            self.get_logger().warn("No scanned data")
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
            self.pub_cmd.publish(self.cmd)
            return 0.0

        dist_left = self.get_sector_min_distance(scan, self.SECTOR_LEFT_MIN, self.SECTOR_LEFT_MAX)
        dist_right = self.get_sector_min_distance(scan, self.SECTOR_RIGHT_MIN, self.SECTOR_RIGHT_MAX)
        dist_front = self.get_sector_min_distance(scan, self.SECTOR_FRONT_MIN, self.SECTOR_FRONT_MAX)

        error = dist_left - dist_right 
        
        p_term = self.Kp * error
        
        self.error_integral += error * self.dt
        MAX_INTEGRAL = 0.5 
        self.error_integral = np.clip(self.error_integral, -MAX_INTEGRAL, MAX_INTEGRAL)
        i_term = self.Ki * self.error_integral
        #np.clip:clip values to an interval

        # derivative term :damping and prediction
        error_derivative = (error - self.previous_error) / self.dt
        d_term = self.Kd * error_derivative
        self.previous_error = error
        
        angular_output = p_term + i_term + d_term
        
        OBSTACLE_THRESHOLD = 0.4
        CORNER_DIST_THRESHOLD = 4.0

        if dist_front < OBSTACLE_THRESHOLD:
            linear_output = 0.0 
            
            if dist_left > dist_right:
                angular_output = 0.5
            else:
                angular_output = -0.5
            
        # corner detection
        elif dist_left > CORNER_DIST_THRESHOLD or dist_right > CORNER_DIST_THRESHOLD:
            
            linear_output = self.TARGET_LINEAR_SPEED * 0.5

        else:
            linear_output = self.TARGET_LINEAR_SPEED
            
        MAX_ANGULAR_Z = 1.5
        self.cmd.angular.z = np.clip(angular_output, -MAX_ANGULAR_Z, MAX_ANGULAR_Z)
        self.cmd.linear.x = linear_output
        
        self.pub_cmd.publish(self.cmd)
        
        return angular_output


    def publish_odom(self, frame_id: str, child_frame_id: str, x: float, y: float, yaw: float, publisher):
        

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
        """Vezérlés, Szimuláció, Odometria, Pontfelhő."""
        
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


def main(args=None):
    rclpy.init(args=args)
    tb3_follower = TurtleBot3WallFollower()
    rclpy.spin(tb3_follower)
    tb3_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()