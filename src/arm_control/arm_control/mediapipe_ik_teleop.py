#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import time
from math import sqrt
import numpy as np

class KalmanFilter1D:
    
    # 1D Kalmann Filter
    def __init__(self, process_noise=1e-3, measurement_noise=1e-2):
        self.x = np.zeros(2)                  # State vector: [position, velocity]
        self.P = np.eye(2) * 1.0              # Covariance matrix (initial uncertainty)
        self.Q = np.eye(2) * process_noise    # Process noise covariance
        self.R = np.array([[measurement_noise]]) # Measurement noise covariance
        self.H = np.array([[1.0, 0.0]])       # Observation matrix (we only see position)
        self.initialized = False

    def update(self, measurement: float, dt: float = 0.05) -> float:
        # State transition: position += velocity * dt
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        # Set first reading to zero
        if not self.initialized:
            self.x[0] = measurement
            self.initialized = True
            return measurement

        # --- Predict ---
        self.x = F @ self.x # matrix multiplication for new position calculation
        self.P = F @ self.P @ F.T + self.Q

        # --- Update ---
        y = np.array([measurement]) - self.H @ self.x     # Residual (difference between prediction and data)
        S = self.H @ self.P @ self.H.T + self.R           # Kalman gain denominator (total uncertainty)
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        
        # New State and Covariance
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0])

from ikpy.chain import Chain

from ament_index_python.packages import get_package_share_directory
import os

package_share_dir = get_package_share_directory('arm_control')

MODEL_FILE = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')

PRINT_DATA = True

base_options = python.BaseOptions(model_asset_path=MODEL_FILE)

# --- Manual Drawing Logic ---
# Defined to avoid using mp.solutions.drawing_utils
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (9, 10), (10, 11), (11, 12),              # Middle
    (13, 14), (14, 15), (15, 16),             # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17)                 # Palm
]

def draw_on_frame(image, result):
    if result is None or not result.hand_landmarks:
        return image
    
    h, w, _ = image.shape
    for hand_landmarks in result.hand_landmarks:
        # Draw connections (lines)
        for connection in HAND_CONNECTIONS:
            start = hand_landmarks[connection[0]]
            end = hand_landmarks[connection[1]]
            cv2.line(image, (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), (0, 255, 0), 2)
        
        # Draw points (circles)
        for lm in hand_landmarks:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)
            
        # Calculate palm center
        wrist = hand_landmarks[0]
        index_mcp = hand_landmarks[5]
        pinky_mcp = hand_landmarks[17] 
        palm_center_x = int(((wrist.x + index_mcp.x + pinky_mcp.x)/3) * w)
        palm_center_y = int(((wrist.y + index_mcp.y + pinky_mcp.y)/3) * h)
        
        # Draw palm center
        cv2.circle(image, (palm_center_x, palm_center_y), 8, (0, 255, 255), -1)
        cv2.putText(image, "Palm", (palm_center_x + 10, palm_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw Index to thumb Grip Length
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        
        pinch_dist = round(sqrt(pow((index_tip.x - thumb_tip.x), 2) + pow((index_tip.y - thumb_tip.y), 2)), 5)
        
        # Draw Camera
        cv2.putText(image, f"Dist: {pinch_dist}", (int(index_tip.x * w - 10), int(index_tip.y * h - 10)), 
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
        cv2.line(image, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 
                     (int(index_tip.x * w), int(index_tip.y * h)), (0, 255, 255), 5)
        
        # Draw Gripper Text
        cv2.putText(image, f"Gripper: {'Closed' if pinch_dist < 0.1 else 'Open'}", (50,50), 
            cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255) if pinch_dist < 0.1 else (0, 255, 0), 5)
        
    return image


def print_landmarks(result, frame_width, frame_height):
    """Prints the pixel coordinates of each landmark to the console."""
    if result is None or not result.hand_landmarks:
        return
    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
        print(f"\n--- Hand {hand_index} ---")
        for idx, lm in enumerate(hand_landmarks):
            pixel_x = int(lm.x * frame_width)
            pixel_y = int(lm.y * frame_height)
            print(f"Point {idx}: (x: {lm.x}, y: {lm.y}, z: {lm.z:.3f})")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def pinch_to_gripper(pinch_dist: float,
                     dist_open: float = 0.15,
                     dist_closed: float = 0.04) -> float:
    t = (pinch_dist - dist_closed) / (dist_open - dist_closed)
    t = max(0.0, min(1.0, t))
    return -1.57 + t * 3.14


class HandTrackerNode(Node):

    def __init__(self):

        super().__init__("hand_tracker_node")

        self.latest_result = None # Global variable to hold the latest detection result

        # --- Initialize Landmarker ---
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            result_callback=self.result_callback
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)

        self.timer = self.create_timer(0.05, self.process_frame)

        # Setup publisher
        self.pub = self.create_publisher(Float64MultiArray, "/arm_forward_controller/commands", 10)

        pkg = get_package_share_directory("arm_description")
        self.urdf_path = os.path.join(pkg, "urdf", "mini_arm.urdf")
        self.get_logger().info(f"Loading URDF: {self.urdf_path}")

        # Build IK chain
        self.chain = Chain.from_urdf_file(self.urdf_path)

        # Commanded joint order (MUST match the controller expecting 6 values)
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

        self.kf_palm_z = KalmanFilter1D(process_noise=1e-3, measurement_noise=1e-2)

        # Joint mapping (flip directions + add offsets)
        self.sign = np.array([-1, -1, 1, 1, -1, -1], dtype=float)
        self.offset = np.array([-0.366519, 0.436332, -0.872665, -1.13446, 1.46608, 0.0], dtype=float)

        # URDF limits (used +/-1.57)
        self.limits_lo = np.array([-1.57] * 6, dtype=float)
        self.limits_hi = np.array([ 1.57] * 6, dtype=float)

        self.print_state()



    def remap(self, value, in_min, in_max, out_min, out_max):
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


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



    
    def result_callback(self, result, output_image, timestamp_ms):
        """Callback function to receive results asynchronously."""
        self.latest_result = result

    def process_frame(self):
        
            success, frame = self.cap.read()
            if not success: return

            # Flip for selfie view and convert BGR to RGB
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Send frame asynchronously
            self.landmarker.detect_async(mp_image, int(time.time() * 1000))

            # Draw the results on the current frame
            frame = draw_on_frame(frame, self.latest_result)
            
            if self.latest_result and self.latest_result.hand_landmarks:

                hand_landmarks = self.latest_result.hand_landmarks[0]

                wrist = hand_landmarks[0]
                index_mcp = hand_landmarks[5]
                pinky_mcp = hand_landmarks[17]
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
        

                # Compute normalized palm center
                palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
                palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3
                palm_z = (wrist.z + index_mcp.z + pinky_mcp.z) / 3

                xmin = 0.1
                xmax = 0.9

                ymin = 0.05
                ymax = 0.95

                zmin = -0.012
                zmax = -0.1 

                # Remapping camera data for solving ik
                palm_z_smooth = self.kf_palm_z.update(palm_z, dt=0.05)

                # camera z-axis -> arm x-axis
                self.target[0] = self.remap(palm_z_smooth, zmin, zmax, -0.1, 0.5)

                # camera x-axis -> arm y-axis (inverted)
                self.target[1] = self.remap(palm_x, xmin, xmax, 0.3, -0.3)

                # camera y-axis -> arm z-axis
                self.target[2] =  self.remap(palm_y, ymax, ymin, 0, 0.3)
                
                self.clamp_target()

                pinch_dist = sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)

                try:

                    q6 = self.solve_ik()
                    self.q = np.array(q6, dtype=float)
                    self.q[5] = pinch_to_gripper(pinch_dist)

                    self.publish(self.q)
                    self.print_state()
                except Exception as e:
                    self.get_logger().error(f"IK failed: {e}")

                if PRINT_DATA:
                    print(palm_z)
                    print(self.target[0])

            cv2.imshow("Hand Tracking", frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = HandTrackerNode()

    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
        

if __name__ == "__main__":
    main()