#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped

from ikpy.chain import Chain


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))



class CartesianIKTeleop(Node):
    def __init__(self):
        super().__init__("mini_arm_cartesian_ik_teleop")

        self.pub = self.create_publisher(Float64MultiArray, "/arm_forward_controller/commands", 10)
        self.sub = self.create_subscription(PoseStamped, "/position_topic", self.target_cb, 10)

        pkg = get_package_share_directory("arm_description")
        self.urdf_path = os.path.join(pkg, "urdf", "mini_arm.urdf")
        self.get_logger().info(f"Loading URDF: {self.urdf_path}")

        # Build IK chain
        self.chain = Chain.from_urdf_file(self.urdf_path)

        # Your commanded joint order (MUST match the controller expecting 6 values)
        self.joint_names = [
            "base_rotator_joint",
            "shoulder_joint",
            "elbow_joint",
            "wrist_joint",
            "end_joint",
            "gear_right_joint",
        ]

        # Activate only those joints inside ikpy chain
        active_mask = [False] * len(self.chain.links)
        for jn in self.joint_names:
            for i, link in enumerate(self.chain.links):
                if link.name == jn:
                    active_mask[i] = True
                    break
        self.chain.active_links_mask = active_mask

        # Target pose (meters) and step
        self.target = np.array([0.00, 0.00, 0.00], dtype=float)
        self.step = 0.01

        # Safety workspace box (meters)
        self.xyz_min = np.array([-0.5, -0.5, -0.5], dtype=float)
        self.xyz_max = np.array([0.5,  0.5, 0.5], dtype=float)

        # Last commanded joints (rad), used as IK initial guess
        self.q = np.zeros(6, dtype=float)

        # Optional joint mapping (flip directions / add offsets)
        # If your 2nd joint needs flipped, set sign[1] = -1
        self.sign = np.array([-1, -1, 1, 1, -1, -1], dtype=float)
        self.offset = np.array([-0.366519, 0.436332, -0.872665, -1.13446, 1.46608, 0.0], dtype=float)

        # URDF limits (you used +/-1.57 everywhere)
        self.limits_lo = np.array([-1.57] * 6, dtype=float)
        self.limits_hi = np.array([ 1.57] * 6, dtype=float)

        self.print_state()



    def remap(self, value, in_min, in_max, out_min, out_max):
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

    def target_cb(self, msg: PoseStamped):
        xmin = 0.1
        xmax = 0.9

        ymin = 0.05
        ymax = 0.95

        zmin = -0.012
        zmax = -0.1

        self.target[0] = self.remap(msg.pose.position.x, xmin, xmax, -0.3, 0.3)
        self.target[1] = self.remap(msg.pose.position.z, zmin, zmax, -0.3, 0.3)
        self.target[2] = self.remap(msg.pose.position.y, ymax, ymin, 0, 0.3)
        self.clamp_target()

        try:
            q6 = self.solve_ik()
            self.q = np.array(q6, dtype=float)
            self.publish(self.q)
            self.print_state()
        except Exception as e:
            self.get_logger().error(f"IK failed: {e}")


    def print_state(self):
        self.get_logger().info(
            f"target xyz: [{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]  step: {self.step:.3f}"
        )

    def clamp_target(self):
        self.target = np.minimum(np.maximum(self.target, self.xyz_min), self.xyz_max)

    def publish(self, q6):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q6]
        self.pub.publish(msg)

    def solve_ik(self):
        # IMPORTANT FIX:
        # ikpy Chain.inverse_kinematics expects a 3D target position vector (x,y,z),
        # NOT a 4x4 matrix.
        target_xyz = np.array(self.target, dtype=float).reshape(3,)

        # Build initial guess vector of length = number of links
        guess = np.zeros(len(self.chain.links), dtype=float)

        # Put current 6 joint values into the guess where those joints exist in the chain
        for j, jn in enumerate(self.joint_names):
            for i, link in enumerate(self.chain.links):
                if link.name == jn:
                    q_urdf_guess = self.sign[j] * float(self.q[j]) + self.offset[j]
                    guess[i] = q_urdf_guess
                    break

        sol = self.chain.inverse_kinematics(target_xyz, initial_position=guess)

        # Extract URDF-space solution in our 6-joint order
        q_urdf = np.zeros(6, dtype=float)
        for j, jn in enumerate(self.joint_names):
            for i, link in enumerate(self.chain.links):
                if link.name == jn:
                     q_urdf[j] = float(sol[i])
                     break

        # Convert URDF-space -> physical command space
        # q_phys = sign * (q_urdf - offset)
        q_phys = self.sign * (q_urdf - self.offset)

        # Clamp physical commands to limits
        for i in range(6):
             q_phys[i] = clamp(float(q_phys[i]), float(self.limits_lo[i]), float(self.limits_hi[i]))

        return q_phys

def main():
    rclpy.init()
    node = CartesianIKTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()

