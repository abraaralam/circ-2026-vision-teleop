# CIRC 2026 Vision Teleop

This ROS 2 workspace was adapted from *Combat-Craft/tasc_arm* to build out a computer vision teleop node for the TASC arm, part of TASC's submission to the Canadian International Rover Competition (CIRC). Mediapipe and OpenCV were integrated from *Felixly1/hand-track-end-effector* to convert hand tracking values to arm movement. This repo currently contains:
- **URDF + meshes** for RViz visualization
- **ros2_control** setup (controllers + hardware plugin scaffold)
- **Teleop** [NEW] vision hand tracking for joint control, plus IK-based Cartesian teleop (ikpy)
- **Bringup launch** to start RViz, robot_state_publisher, ros2_control, and joystick nodes
<img width="600"  alt="move arm gif" src="https://github.com/user-attachments/assets/199e35af-d255-4c68-9de5-02e585cfb7de" />


## Pipeline
The system follows a six-stage pipeline to translate visual data into physical joint commands:
- Frame Capture: OpenCV handles asynchronous webcam polling to maintain high framerates.
- Landmark Calculation: MediaPipe extracts 21 3D landmarks. The Palm Center is derived by averaging the coordinates of the Wrist (0), Index MCP (5), and Pinky MCP (17).
- Kalman Filtering: To eliminate jitter in the Z-axis (depth), a 1D Kalman filter smooths the signal.
- Workspace Mapping: Coordinates are remapped from the camera's FOV to the robot's physical reach limits.
- Inverse Kinematics (IK): The ikpy library solves for the 6-DOF joint angles required to reach the target (x, y, z).
- Joint Command Publishing: Commands are sent to `/arm_forward_controller/commands` as a `Float64MultiArray`.

### Kalman Filter Logic
The z-axis data from MediaPipe was particularly susceptible to noise and resulted in jitter. To handle sensor noise, a Kalman filter was implemented. 

The filter predicts the next state based on velocity and updates it using the predicted and actual measurement difference. For a given state with position and velocity:
```math
\begin{bmatrix} pos_{new} \\ vel_{new} \end{bmatrix} = \begin{bmatrix} 1 & dt \\ 0 & 1 \end{bmatrix} \begin{bmatrix} pos_{old} \\ vel_{old} \end{bmatrix}
```
This is the same thing as saying $`pos_{new}=pos_{old}+vel_{old}\cdot dt`$, and is how we calculate the smoothened z-axis position from the camera.

Kalman Gain ($K$):
```math
K=\frac{P \cdot H^T}{H \cdot P \cdot H^T + R}
```
Where $R$ represents measurement noise. A higher $R$ (distrust in camera tracking values) results in a smaller $K$, leading to less reactive arm movement.

## Modified Repo Layout
```bash
/tasc_arm/src/
├── arm_description/ # URDF + STL meshes
├── arm_bringup/ # launch files (RViz + control bringup)
├── arm_control/ # ros2_control configs + hardware plugin + teleop nodes
├──── arm_control/
│     └── mediapipe_ik_teleop.py # **[NEW]** vision teleop node
├──── models/ # **[NEW]** hand tracking model
│     └── hand_landmarker.task
└──── CMakeLists.txt # updated
```
### Packages

#### `arm_description`
- Contains the robot model used by RViz and robot_state_publisher.

#### `arm_control`
- Contains controller config + teleop nodes.
- Also includes a **ros2_control hardware interface plugin** scaffold:

#### `arm_bringup`
- Launch files to bring up the full stack.
- Key file:
  - `src/arm_bringup/launch/bringup.launch.py`

## Dependencies

Assuming Ubuntu 22.04 + ROS 2 Humble.

Install common deps:
```bash
sudo apt update
sudo apt install -y \
  ros-humble-rviz2 \
  ros-humble-robot-state-publisher \
  ros-humble-controller-manager \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-forward-command-controller \
  ros-humble-joy
```
Python deps (for IK teleop):
```bash
python3 -m pip install --user ikpy numpy mediapipe opencv-python
```


## Build & Launch
From the workspace root:

```bash
cd ~/tasc_arm
source /opt/ros/humble/setup.bash
colcon build --packages-select arm_control arm_description arm_bringup --symlink-install
source install/setup.bash

# Terminal 1: RViz Simulation
ros2 launch arm_bringup bringup.launch.py

# This might take a while. If it tells you to either force
# quit or wait for the program, WAIT. Eventually, the
# simulation will load.

# Terminal 2: Hand Tracking Node
ros2 run arm_control mediapipe_ik_teleop.py
```

## Limitations
The RViz simulation does not fully load the gripper from the URDF, so pinching is only visualized with one side of the gripper claw.

<img width="600"  alt="pinch gif" src="https://github.com/user-attachments/assets/276da513-8d0d-49f6-94da-1af8b93d18ea" />

The new TASC arm in *Monti-1/arm_ws* has a fully loaded URDF, which could potentially be implemented to fix the simulation issue.

