#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import numpy as np
import torch

# Import from the fabrics_sim package
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel


NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16


class KukaFabricPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("kuka_fabric_publisher", anonymous=True)

        # ROS msgs
        self.iiwa_joint_state = None
        self.allegro_joint_state = None

        # Create a ROS publisher
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegro/joint_cmd", JointState, queue_size=10
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegro/joint_states", JointState, self.allegro_joint_state_callback
        )

        # ROS rate
        self.rate = rospy.Rate(60)  # 60 Hz

        # Number of environments (batch size) and device setup
        self.num_envs = 1  # Single environment for this example
        self.device = "cuda:0"

        # Time step
        self.control_dt = 1.0 / 60.0

        # Setup the Fabric
        self._setup_fabric_action_space()

        self.loop_count = 0

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_STATE = False
        if not self.WAIT_FOR_ALLEGRO_STATE:
            rospy.logwarn("NOT WAITING FOR ALLEGRO STATE")
            self.allegro_joint_state = np.zeros(NUM_HAND_JOINTS)

        while not rospy.is_shutdown():
            if (
                self.iiwa_joint_state is not None
                and self.allegro_joint_state is not None
            ):
                rospy.loginfo("Got iiwa and allegro joint states")
                break

            rospy.loginfo(
                f"Waiting: iiwa_joint_state: {self.iiwa_joint_state}, allegro_joint_state: {self.allegro_joint_state}"
            )
            rospy.sleep(0.1)

        # VERY IMPORTANT: Set the initial fabric_q to match the initial joint states
        assert self.iiwa_joint_state is not None
        assert self.allegro_joint_state is not None
        self.fabric_q.copy_(
            torch.tensor(
                np.concatenate(
                    [self.iiwa_joint_state, self.allegro_joint_state], axis=0
                ),
                device=self.device,
            ).unsqueeze(0)
        )

    def iiwa_joint_state_callback(self, msg: JointState) -> None:
        self.iiwa_joint_state = np.array(msg.position)

    def allegro_joint_state_callback(self, msg: JointState) -> None:
        self.allegro_joint_state = np.array(msg.position)

    def _setup_fabric_action_space(self):
        # Initialize warp
        initialize_warp(warp_cache_name="")

        # Set up the world model
        THICKNESS = 0.02
        HEIGHT = 1
        self.fabric_world_dict = {
            "right_wall": {
                "env_index": "all",
                "type": "box",
                "scaling": f"0.9 {THICKNESS} {HEIGHT}",
                "transform": f"0 0.37 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
            },
            "left_wall": {
                "env_index": "all",
                "type": "box",
                "scaling": f"0.9 {THICKNESS} {HEIGHT}",
                "transform": f"0 -0.37 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
            },
            "back_wall": {
                "env_index": "all",
                "type": "box",
                "scaling": f"{THICKNESS} 0.74 {HEIGHT}",
                "transform": f"-0.45 0 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
            },
            "front_wall": {
                "env_index": "all",
                "type": "box",
                "scaling": f"{THICKNESS} 0.74 {HEIGHT}",
                "transform": f"0.45 0 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
            },
            "table": {
                "env_index": "all",
                "type": "box",
                "scaling": f"0.45 0.74 {THICKNESS}",
                "transform": "0.225 0 0 0 0 0 1",  # x y z qx qy qz qw
            },
            "ceiling": {
                "env_index": "all",
                "type": "box",
                "scaling": f"0.9 0.74 {THICKNESS}",
                "transform": f"0 0 {HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
            },
        }
        self.fabric_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=20,
            device=self.device,
            world_dict=self.fabric_world_dict,
        )
        self.fabric_object_ids, self.fabric_object_indicator = (
            self.fabric_world_model.get_object_ids()
        )

        # Create Kuka-Allegro Pose Fabric
        self.fabric = KukaAllegroPoseFabric(
            batch_size=self.num_envs,
            device=self.device,
            timestep=self.control_dt,
            graph_capturable=True,
        )
        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        # Initialize random targets for palm and hand
        self.fabric_hand_target = self.sample_hand_target()
        self.fabric_palm_target = self.sample_palm_target()

        # Joint states
        self.fabric_q = torch.zeros(
            self.num_envs, self.fabric.num_joints, device=self.device
        )
        self.fabric_qd = torch.zeros_like(self.fabric_q)
        self.fabric_qdd = torch.zeros_like(self.fabric_q)

        # Capture the fabric graph for CUDA optimization
        fabric_inputs = [
            self.fabric_hand_target,
            self.fabric_palm_target,
            "euler_zyx",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.fabric_object_ids,
            self.fabric_object_indicator,
        ]
        (
            self.fabric_cuda_graph,
            self.fabric_q_new,
            self.fabric_qd_new,
            self.fabric_qdd_new,
        ) = capture_fabric(
            fabric=self.fabric,
            q=self.fabric_q,
            qd=self.fabric_qd,
            qdd=self.fabric_qdd,
            timestep=self.control_dt,
            fabric_integrator=self.fabric_integrator,
            inputs=fabric_inputs,
            device=self.device,
        )

    def sample_palm_target(self) -> torch.Tensor:
        # x forward, y left, z up
        # roll pitch yaw
        palm_target = torch.zeros(self.num_envs, 6, device="cuda", dtype=torch.float)
        palm_target[:, 0] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(0, 0.5)
            .to(device=palm_target.device, dtype=palm_target.dtype)
        )
        palm_target[:, 1] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(-0.4, 0.4)
            .to(device=palm_target.device, dtype=palm_target.dtype)
        )
        palm_target[:, 2] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(0.3, 0.5)
            .to(device=palm_target.device, dtype=palm_target.dtype)
        )
        palm_target[:, 3:6] = (
            torch.FloatTensor(self.num_envs, 3)
            .uniform_(0, 2 * np.pi)
            .to(device=palm_target.device, dtype=palm_target.dtype)
        )
        return palm_target

    def sample_hand_target(self) -> torch.Tensor:
        # Joint limits for the hand and palm targets
        fabric_hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
        )
        fabric_hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
        )

        # Initialize random targets for palm and hand
        fabric_hand_target = (fabric_hand_maxs - fabric_hand_mins) * torch.rand(
            self.num_envs, fabric_hand_maxs.numel(), device=self.device
        ) + fabric_hand_mins
        return fabric_hand_target

    def run(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update fabric targets for palm and hand
            SWITCH_TARGET_FREQ = 120
            if self.loop_count % SWITCH_TARGET_FREQ == 0:
                self.fabric_palm_target.copy_(self.sample_palm_target())
                self.fabric_hand_target.copy_(self.sample_hand_target())

            # Step the fabric using the captured CUDA graph
            self.fabric_cuda_graph.replay()
            self.fabric_q.copy_(self.fabric_q_new)
            self.fabric_qd.copy_(self.fabric_qd_new)
            self.fabric_qdd.copy_(self.fabric_qdd_new)

            # Prepare a JointState message for ROS
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = rospy.Time.now()
            allegro_msg = JointState()
            allegro_msg.header.stamp = rospy.Time.now()

            # Set joint values
            iiwa_msg.name = [
                "iiwa_joint_1",
                "iiwa_joint_2",
                "iiwa_joint_3",
                "iiwa_joint_4",
                "iiwa_joint_5",
                "iiwa_joint_6",
                "iiwa_joint_7",
            ]
            allegro_msg.name = [
                "allegro_joint_0",
                "allegro_joint_1",
                "allegro_joint_2",
                "allegro_joint_3",
                "allegro_joint_4",
                "allegro_joint_5",
                "allegro_joint_6",
                "allegro_joint_7",
                "allegro_joint_8",
                "allegro_joint_9",
                "allegro_joint_10",
                "allegro_joint_11",
                "allegro_joint_12",
                "allegro_joint_13",
                "allegro_joint_14",
                "allegro_joint_15",
            ]

            iiwa_msg.position = (
                self.fabric_q.cpu().numpy()[0, :NUM_ARM_JOINTS].tolist()
            )  # Use the joint positions from the fabric
            iiwa_msg.velocity = (
                self.fabric_qd.cpu().numpy()[0, :NUM_ARM_JOINTS].tolist()
            )  # Velocities from fabric
            iiwa_msg.effort = [
                0.0
            ] * NUM_ARM_JOINTS  # Set efforts to zero for simplicity
            allegro_msg.position = (
                self.fabric_q.cpu()
                .numpy()[0, NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS]
                .tolist()
            )  # Use the joint positions from the fabric
            allegro_msg.velocity = (
                self.fabric_qd.cpu()
                .numpy()[0, NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS]
                .tolist()
            )
            allegro_msg.effort = [
                0.0
            ] * NUM_HAND_JOINTS  # Set efforts to zero for simplicity

            # Publish the joint states
            self.iiwa_cmd_pub.publish(iiwa_msg)
            self.allegro_cmd_pub.publish(allegro_msg)

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                f"Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz"
            )

            self.loop_count += 1


if __name__ == "__main__":
    try:
        kuka_fabric_publisher = KukaFabricPublisher()
        kuka_fabric_publisher.run()
    except rospy.ROSInterruptException:
        pass
