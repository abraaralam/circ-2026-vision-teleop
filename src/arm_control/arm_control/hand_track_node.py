#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import mediapipe as mp
import time
from math import sqrt
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ament_index_python.packages import get_package_share_directory
import os


package_share_dir = get_package_share_directory('hand_tracker')
MODEL_FILE = os.path.join(package_share_dir, 'hand_landmarker.task') # Path to your hand landmark model
PRINT_DATA = False  # Set to False to stop printing to console
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
        #Draw connections (lines)
        for connection in HAND_CONNECTIONS:
            start = hand_landmarks[connection[0]]
            end = hand_landmarks[connection[1]]
            cv2.line(image, (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), (0, 255, 0), 2)
        
        #Draw points (circles)
        for lm in hand_landmarks:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)
            
        #Calculate palm center
        wrist = hand_landmarks[0]
        index_mcp = hand_landmarks[5]
        pinky_mcp = hand_landmarks[17] 
        palm_center_x = int(((wrist.x + index_mcp.x + pinky_mcp.x)/3) * w)
        palm_center_y = int(((wrist.y + index_mcp.y + pinky_mcp.y)/3) * h)
        
        #Draw palm center
        cv2.circle(image, (palm_center_x, palm_center_y), 8, (0, 255, 255), -1)
        cv2.putText(image, "Palm", (palm_center_x + 10, palm_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        #Draw Index to thumb Grip Length
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        pinch_dist = round(sqrt(pow((index_tip.x - thumb_tip.x), 2) + pow((index_tip.y - thumb_tip.y), 2)), 5)
        
        #Draw Camera
        cv2.putText(image, f"Dist: {pinch_dist}", (int(index_tip.x * w - 10), int(index_tip.y * h - 10)), 
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
        cv2.line(image, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 
                     (int(index_tip.x * w), int(index_tip.y * h)), (0, 255, 255), 5)
        
        #Draw Gripper Text
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


class HandTrackerNode(Node):

    def __init__(self):
        super().__init__("hand_tracker_node")

        self.pos_pub = self.create_publisher(PoseStamped, "/position_topic", 10)

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

                # Compute normalized palm center
                palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
                palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3
                palm_z = (wrist.z + index_mcp.z + pinky_mcp.z) / 3

                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = "camera_frame"

                pose.pose.position.x = float(palm_x)
                pose.pose.position.y = float(palm_y)
                pose.pose.position.z = float(palm_z)

                self.pos_pub.publish(pose)

                self.get_logger().info(f"Sent pose: x:{pose.pose.position.x}, y:{pose.pose.position.y}, z: {pose.pose.position.z}")

                if PRINT_DATA:
                    print(palm_x, palm_y, palm_z)

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