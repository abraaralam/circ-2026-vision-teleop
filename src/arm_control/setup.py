from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'arm_control'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),

    scripts=glob('arm_control/*.py'),

    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), ['config/controllers.yaml']),
        (os.path.join('share', package_name), ['plugin.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmedtabl',
    maintainer_email='ahmedmegahed20142@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_teleop = arm_control.keyboard_teleop:main',
            'keyboard_cartesian_ik_teleop = arm_control.keyboard_cartesian_ik_teleop:main',
            'ps5_arm_teleop = arm_control.ps5_arm_teleop:main',
            'ps5_cartesian_ik_teleop = arm_control.ps5_cartesian_ik_teleop:main',
            'logitech_joystick_ik = arm_control.logitech_joystick_ik:main',
            'mediapipe_ik_teleop = arm_control.mediapipe_ik_teleop:main',
        ],
    },
)