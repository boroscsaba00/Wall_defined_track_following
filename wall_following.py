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

        self.Kp = 1.5 
        self.Ki = 0.001
        self.Kd = 0.5
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.target_distance = 0.35 

        
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


    def calculate_control(self):
        """Falkövetés PID vezérlővel."""
        scan = self.scan_data
        
        if not scan.ranges:
            
            return 0.0

        
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        
        idx_A = int((90 * math.pi/180 - scan.angle_min) / scan.angle_increment)
        idx_B = int((100 * math.pi/180 - scan.angle_min) / scan.angle_increment)

        idx_A = max(0, min(len(scan.ranges) - 1, idx_A))
        idx_B = max(0, min(len(scan.ranges) - 1, idx_B))
        
        dist_A = scan.ranges[idx_A]
        dist_B = scan.ranges[idx_B]

        
        current_distance = (dist_A + dist_B) / 2.0 if dist_A < scan.range_max and dist_B < scan.range_max else self.target_distance

        error = self.target_distance - current_distance

        self.error_integral += error * self.dt
        error_derivative = (error - self.previous_error) / self.dt
        
        angular_output = self.Kp * error + self.Ki * self.error_integral + self.Kd * error_derivative
        self.previous_error = error

        idx_front = int((0 - scan.angle_min) / scan.angle_increment)
        dist_front = scan.ranges[idx_front] if scan.ranges[idx_front] < scan.range_max else 10.0

        if dist_front < 0.4: 
            linear_output = 0.0
            angular_output = 0.5
        else:
            linear_output = 0.1 

        
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