#!/usr/bin/env python

import numpy as np
import rospy
import torch

# Import from the fabrics_sim package
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import capture_fabric, initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from fabric_world import world_dict_robot_frame

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16


class IiwaAllegroFabricPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("iiwa_allegro_fabric_publisher")

        # ROS msgs
        self.iiwa_joint_state = None
        self.allegro_joint_state = None
        self.palm_target = None
        self.hand_target = None

        # Publisher and subscriber
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.palm_target_sub = rospy.Subscriber(
            "/palm_target", Float64MultiArray, self.palm_target_callback
        )
        self.hand_target_sub = rospy.Subscriber(
            "/hand_target", Float64MultiArray, self.hand_target_callback
        )
        self.fabric_pub = rospy.Publisher("/fabric_state", JointState, queue_size=10)

        # ROS rate
        self.rate = rospy.Rate(60)  # 60 Hz
        self.device = "cuda:0"

        # Time step
        self.control_dt = 1.0 / 60.0

        # Setup the Fabric
        self._setup_fabric_action_space()

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_STATE = True
        if not self.WAIT_FOR_ALLEGRO_STATE:
            rospy.logwarn("NOT WAITING FOR ALLEGRO STATE")
            self.allegro_joint_state = np.zeros(NUM_HAND_JOINTS)

        # Wait for the initial joint states
        while not rospy.is_shutdown():
            if (
                self.iiwa_joint_state is not None
                and self.allegro_joint_state is not None
            ):
                rospy.loginfo("Got iiwa and allegro joint states")
                break

            rospy.loginfo(
                f"Waiting: iiwa_joint_state: {self.iiwa_joint_state}, allegro_joint_state: {self.allegro_joint_state}, palm_target: {self.palm_target}, hand_target: {self.hand_target}"
            )
            rospy.sleep(0.1)

        # VERY IMPORTANT: Set the initial fabric_q to match the initial joint states
        assert self.iiwa_joint_state is not None
        assert self.allegro_joint_state is not None
        self.fabric_q.copy_(
            torch.from_numpy(
                np.concatenate(
                    [self.iiwa_joint_state, self.allegro_joint_state], axis=0
                ),
            )
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

    def iiwa_joint_state_callback(self, msg: JointState) -> None:
        self.iiwa_joint_state = np.array(msg.position)

    def allegro_joint_state_callback(self, msg: JointState) -> None:
        self.allegro_joint_state = np.array(msg.position)

    def palm_target_callback(self, msg: Float64MultiArray) -> None:
        self.palm_target = np.array(msg.data)

    def hand_target_callback(self, msg: Float64MultiArray) -> None:
        self.hand_target = np.array(msg.data)

    def _setup_fabric_action_space(self):
        self.num_envs = 1  # Single environment for this example

        # Initialize warp
        initialize_warp(warp_cache_name="")

        # Set up the world model
        self.fabric_world_dict = world_dict_robot_frame
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
        self.fabric_hand_target = torch.zeros(
            self.num_envs, 5, device="cuda", dtype=torch.float
        )
        self.fabric_palm_target = torch.zeros(
            self.num_envs, 6, device="cuda", dtype=torch.float
        )

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

    def run(self):
        # Must have initial joint states before starting
        # Do not need to have targets yet
        assert self.iiwa_joint_state is not None
        assert self.allegro_joint_state is not None

        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            if self.palm_target is not None and self.hand_target is not None:
                # Step fabric with the targets
                # Update fabric targets for palm and hand
                self.fabric_palm_target.copy_(
                    torch.from_numpy(self.palm_target).unsqueeze(0).float().to(self.device)
                )
                self.fabric_hand_target.copy_(
                    torch.from_numpy(self.hand_target).unsqueeze(0).float().to(self.device)
                )

                # Step the fabric using the captured CUDA graph
                self.fabric_cuda_graph.replay()
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)
            else:
                rospy.logwarn(
                    f"Waiting for targets... palm_target: {self.palm_target}, hand_target: {self.hand_target}"
                )

            # Still publish the joint states even if the targets are not received

            # Prepare a JointState message for ROS
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = rospy.Time.now()
            allegro_msg = JointState()
            allegro_msg.header.stamp = rospy.Time.now()
            fabric_msg = JointState()
            fabric_msg.header.stamp = rospy.Time.now()

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
            fabric_msg.name = iiwa_msg.name + allegro_msg.name

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

            # Leave velocities and efforts as empty lists
            allegro_msg.velocity = []
            allegro_msg.effort = []

            fabric_msg.position = self.fabric_q.cpu().numpy()[0].tolist()
            fabric_msg.velocity = self.fabric_qd.cpu().numpy()[0].tolist()
            fabric_msg.effort = self.fabric_qdd.cpu().numpy()[0].tolist()

            # Publish the joint states
            self.iiwa_cmd_pub.publish(iiwa_msg)
            self.allegro_cmd_pub.publish(allegro_msg)
            self.fabric_pub.publish(fabric_msg)

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                f"Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz"
            )


if __name__ == "__main__":
    try:
        iiwa_fabric_publisher = IiwaAllegroFabricPublisher()
        iiwa_fabric_publisher.run()
    except rospy.ROSInterruptException:
        pass
