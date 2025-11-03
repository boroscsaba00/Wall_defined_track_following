from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_path = get_package_share_directory('wall_track')
    world_path = os.path.join(pkg_path, 'worlds', 'my_world.world')

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_path, '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        # ide jöhetnek további node-ok is
    ])
