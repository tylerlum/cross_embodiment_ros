import copy
import datetime

import numpy as np
import rospy
from sensor_msgs.msg import JointState


class RecordJointData:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("record_joint_data")

        # Initialize storage for latest joint state and command
        self.latest_joint_state = None
        self.latest_joint_cmd = None

        # Initialize lists to store joint states and commands
        self.joint_states = []
        self.joint_cmds = []

        # Create subscribers to both topics
        self.sub_joint_state = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.joint_state_callback
        )
        self.sub_joint_cmd = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.joint_cmd_callback
        )

    def joint_state_callback(self, msg: JointState) -> None:
        """Callback function for joint state messages."""
        self.latest_joint_state = np.array(msg.position)

    def joint_cmd_callback(self, msg: JointState) -> None:
        """Callback function for joint command messages."""
        self.latest_joint_cmd = np.array(msg.position)

    def record_data(self) -> None:
        """Main loop to record data at 60Hz."""
        rate = rospy.Rate(60)  # 60Hz
        rospy.loginfo("Waiting for both joint states and commands to be available...")

        # Wait until both joint state and command are available
        while not rospy.is_shutdown():
            if (
                self.latest_joint_state is not None
                and self.latest_joint_cmd is not None
            ):
                rospy.loginfo(
                    "Both joint states and commands are available, starting to record data..."
                )
                break
            rospy.sleep(0.1)

        # Start recording
        while not rospy.is_shutdown():
            self.joint_states.append(copy.deepcopy(self.latest_joint_state))
            self.joint_cmds.append(copy.deepcopy(self.latest_joint_cmd))
            rate.sleep()

    def save_data(self) -> None:
        """Save recorded data as numpy arrays with timestamped filenames."""
        # Get the current datetime string for filenames
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert lists to numpy arrays
        joint_states_np = np.array(self.joint_states)
        joint_cmds_np = np.array(self.joint_cmds)

        # Define filenames
        joint_states_filename = f"{datetime_str}_joint_states.npy"
        joint_cmds_filename = f"{datetime_str}_joint_cmds.npy"

        # Save the arrays to disk
        np.save(joint_states_filename, joint_states_np)
        np.save(joint_cmds_filename, joint_cmds_np)

        rospy.loginfo(f"Joint states saved to {joint_states_filename}")
        rospy.loginfo(f"Joint commands saved to {joint_cmds_filename}")

    def run(self):
        try:
            # Start recording
            self.record_data()
        except rospy.ROSInterruptException:
            pass
        finally:
            # Save data when node shuts down
            self.save_data()


if __name__ == "__main__":
    recorder = RecordJointData()
    recorder.run()
