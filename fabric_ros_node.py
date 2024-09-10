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
        rospy.init_node('kuka_fabric_publisher', anonymous=True)

        # State
        self.iiwa_joint_state = None
        self.allegro_joint_state = None

        # Create a ROS publisher
        self.iiwa_cmd_pub = rospy.Publisher('/iiwa/joint_cmd', JointState, queue_size=10)
        self.iiwa_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self.iiwa_joint_state_callback)
        self.allegro_cmd_pub = rospy.Publisher('/allegro/joint_cmd', JointState, queue_size=10)
        self.allegro_sub = rospy.Subscriber('/allegro/joint_states', JointState, self.allegro_joint_state_callback)

        # ROS rate
        self.rate = rospy.Rate(60)  # 60 Hz

        # Number of environments (batch size) and device setup
        self.num_envs = 1  # Single environment for this example
        self.device = 'cuda:0'

        # Time step
        self.control_dt = 1.0 / 60.0

        # Setup the Fabric
        self._setup_fabric_action_space()

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_STATE = False
        if not self.WAIT_FOR_ALLEGRO_STATE:
            rospy.logwarn("NOT WAITING FOR ALLEGRO STATE")
            self.allegro_joint_state = np.zeros(NUM_HAND_JOINTS)

        while not rospy.is_shutdown():
            if self.iiwa_joint_state is not None and self.allegro_joint_state is not None:
                rospy.loginfo("Got iiwa and allegro joint states")
                break

            rospy.loginfo(
                f"Waiting: iiwa_joint_state: {self.iiwa_joint_state}, allegro_joint_state: {self.allegro_joint_state}"
            )
            rospy.sleep(0.1)

    def iiwa_joint_state_callback(self, msg: JointState) -> None:
        self.iiwa_joint_state = np.array(msg.position)

    def allegro_joint_state_callback(self, msg: JointState) -> None:
        self.allegro_joint_state = np.array(msg.position)

    def _setup_fabric_action_space(self):
        # Initialize warp
        initialize_warp(warp_cache_name="")

        # Set up the world model
        self.fabric_world_dict = {
            "table": {
                "env_index": "all",
                "type": "box",
                "scaling": "1.0 1.0 0.1",
                "transform": "0 0 0 0 0 0 1",
            },
        }
        self.fabric_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=20,
            device=self.device,
            world_dict=self.fabric_world_dict,
        )
        self.fabric_object_ids, self.fabric_object_indicator = self.fabric_world_model.get_object_ids()

        # Create Kuka-Allegro Pose Fabric
        self.fabric = KukaAllegroPoseFabric(
            batch_size=self.num_envs,
            device=self.device,
            timestep=self.control_dt,
            graph_capturable=True
        )
        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        # Joint limits for the hand and palm targets
        self.fabric_hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
        )
        self.fabric_hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
        )
        self.fabric_palm_mins = torch.tensor(
            [-1.0, -0.7, 0.2, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.fabric_palm_maxs = torch.tensor(
            [0.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )

        # Initialize random targets for palm and hand
        self.fabric_hand_target = (self.fabric_hand_maxs - self.fabric_hand_mins) * torch.rand(
            self.num_envs, self.fabric_hand_maxs.numel(), device=self.device
        ) + self.fabric_hand_mins

        default_palm_target = np.array([-0.6868, 0.0320, 0.6685, -2.3873, -0.0824, 3.1301])
        self.fabric_palm_target = torch.from_numpy(default_palm_target).float().to(self.device).unsqueeze(dim=0).repeat_interleave(self.num_envs, dim=0)

        # Joint states
        self.fabric_q = torch.zeros(self.num_envs, self.fabric.num_joints, device=self.device)
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
        (self.fabric_cuda_graph, self.fabric_q_new, self.fabric_qd_new, self.fabric_qdd_new) = capture_fabric(
            fabric=self.fabric,
            q=self.fabric_q,
            qd=self.fabric_qd,
            qdd=self.fabric_qdd,
            timestep=self.control_dt,
            fabric_integrator=self.fabric_integrator,
            inputs=fabric_inputs,
            device=self.device
        )

    def publish_joint_states(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update fabric targets for palm and hand
            self.fabric_palm_target.copy_(
                (torch.rand(self.num_envs, 6, device=self.device) * (self.fabric_palm_maxs - self.fabric_palm_mins)) + self.fabric_palm_mins
            )
            self.fabric_hand_target.copy_(
                (torch.rand(self.num_envs, 5, device=self.device) * (self.fabric_hand_maxs - self.fabric_hand_mins)) + self.fabric_hand_mins
            )

            # Step the fabric using the captured CUDA graph
            self.fabric_cuda_graph.replay()
            self.fabric_q.copy_(self.fabric_q_new)
            self.fabric_qd.copy_(self.fabric_qd_new)
            self.fabric_qdd.copy_(self.fabric_qdd_new)

            # Prepare a JointState message for ROS
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = rospy.Time.now()

            # Example joint names for KUKA iiwa
            joint_state_msg.name = ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 
                                    'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']
            joint_state_msg.position = self.fabric_q.cpu().numpy()[0].tolist()  # Use the joint positions from the fabric
            joint_state_msg.velocity = self.fabric_qd.cpu().numpy()[0].tolist()  # Velocities from fabric
            joint_state_msg.effort = [0.0] * 7  # Set efforts to zero for simplicity

            # Publish the joint states
            self.iiwa_cmd_pub.publish(joint_state_msg)

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(f"Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz")

if __name__ == '__main__':
    try:
        kuka_fabric_publisher = KukaFabricPublisher()
        kuka_fabric_publisher.publish_joint_states()
    except rospy.ROSInterruptException:
        pass
