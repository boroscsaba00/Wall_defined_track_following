from setuptools import find_packages, setup

package_name = 'projektmunka'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/wall.launch.xml']),
        ('share/' + package_name + '/rviz', ['rviz/wall_follower.rviz'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mgm',
    maintainer_email='doba.daniel@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_scan = projektmunka.can_tf:main",
            "wall_following = projektmunka.wall_following:main",
        ],
    },
)
