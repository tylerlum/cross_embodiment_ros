# cross_embodiment_ros

ROS Noetic code for cross embodiment gap project

Real Nodes:
* `fabric_ros_node.py`: Subscribes to `/iiwa/joint_states` to initialize fabric, subscribes to `/palm_target` and `/hand_target`, publishes to `/iiwa/joint_cmd`
* `rl_policy_node.py`: Subscribes to `/iiwa/joint_states`, `/iiwa/joint_cmd`, and `/object_pose`. Publishes to `/palm_target` and `/hand_target`
* `visualization_ros_node.py`: Subscribes to `/iiwa/joint_states`, `/iiwa/joint_cmd`, `/palm_target`, and `/object_pose`. Visualizes all these things for debugging

Dummy Nodes for testing:
* `fake_robot_ros_node.py`: Pretends to be real robot. Publishes `/iiwa/joint_states` and subscribes to `/iiwa/joint_cmd`
* `fake_policy_ros_node.py`: Pretends to be RL policy. Publishes `/palm_target` and `/hand_target`

```mermaid
graph LR
    D[Perception] -->|/object_pose| A
    A[RL Policy] -->|/hand_target| B[Fabric]
    A -->|/palm_target| B
    B -->|/iiwa/joint_cmd| C[Robot]
    C -->|/iiwa/joint_states| B
    C -->|/iiwa/joint_states| A

    style A fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style B fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style C fill:#bfb,stroke:#333,stroke-width:2px,color:#000
    style D fill:#ffd,stroke:#333,stroke-width:2px,color:#000
```

```mermaid
graph LR
    Camera[Camera] -->|/camera/color/image_raw| SAM[SAM2]
    Camera -->|/camera/color/image_raw| FP[FoundationPose]
    Camera -->|/camera/aligned_depth_to_color/image_raw| FP
    SAM -->|/sam2_mask| FPE[FoundationPoseEvaluator]
    SAM -->|/sam2_mask| FP
    FP -->|/object_pose| FPE
    FP -->|/object_pose| RL[RL Policy]
    FPE -->|/reset| FP

    style Camera fill:#ffd,stroke:#333,stroke-width:2px,color:#000
    style SAM fill:#ffd,stroke:#333,stroke-width:2px,color:#000
    style FP fill:#ffd,stroke:#333,stroke-width:2px,color:#000
    style FPE fill:#ffd,stroke:#333,stroke-width:2px,color:#000
    style RL fill:#f9f,stroke:#333,stroke-width:2px,color:#000
```

```mermaid
graph LR
    O[Other] -->|"/iiwa/joint_states"| V[Visualization]
    O -->|"/iiwa/joint_cmd"| V
    O -->|"/palm_target"| V
    O -->|"/object_pose"| V

    style O fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style V fill:#ffd,stroke:#333,stroke-width:2px,color:#000
```
