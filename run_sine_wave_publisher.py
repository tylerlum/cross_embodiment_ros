import rospy
import copy
import numpy as np
from sensor_msgs.msg import JointState
import math

INIT_JOINT_POS = None


def joint_state_callback(msg: JointState) -> None:
    global INIT_JOINT_POS
    if INIT_JOINT_POS is not None:
        return

    INIT_JOINT_POS = np.array(msg.position)
    rospy.loginfo(f"Initial joint positions: {INIT_JOINT_POS}")


def get_initial_joint_pos() -> np.ndarray:
    sub = rospy.Subscriber("/iiwa/joint_states", JointState, joint_state_callback)
    while INIT_JOINT_POS is None:
        rospy.loginfo("Waiting for INIT_JOINT_POS")
        rospy.sleep(0.1)
        if rospy.is_shutdown():
            raise Exception("rospy shutdown")
    rospy.loginfo("Got INIT_JOINT_POS")
    return INIT_JOINT_POS


def publish_joint_cmd(init_joint_pos: np.ndarray) -> None:
    pub = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=10)
    rate = rospy.Rate(60)

    joint_state_msg = JointState()
    joint_state_msg.header.stamp = rospy.Time.now()
    joint_state_msg.header.frame_id = ""

    joint_state_msg.name = [
        "iiwa_joint_1",
        "iiwa_joint_2",
        "iiwa_joint_3",
        "iiwa_joint_4",
        "iiwa_joint_5",
        "iiwa_joint_6",
        "iiwa_joint_7",
    ]

    assert (
        len(init_joint_pos) == 7
    ), f"Initial joint state must have 7 elements, has {len(init_joint_pos)}"

    # Initial positions (rest of the joints remain the same)
    joint_state_msg.position = copy.deepcopy(init_joint_pos.tolist())

    # Set velocities to 0
    joint_state_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Leave effort empty
    joint_state_msg.effort = []

    # Timer for oscillation
    start_time = rospy.Time.now()
    oscillation_duration = 1.0  # seconds for one full oscillation

    STATIONARY_TIME = 2
    last_publish_time = rospy.Time.now()
    while not rospy.is_shutdown():
        # Calculate elapsed time in seconds
        elapsed_time = (rospy.Time.now() - start_time).to_sec()

        # Calculate the position using a sinusoidal function
        # This gives a smooth oscillation between -0.1 and 0.1
        joint_state_msg.position = copy.deepcopy(init_joint_pos.tolist())
        if elapsed_time > STATIONARY_TIME:
            joint_state_msg.position[0] += 0.1 * math.sin(
                2 * math.pi * (elapsed_time - STATIONARY_TIME) / oscillation_duration
            )
        else:
            rospy.loginfo(
                f"Holding at stationary position for {STATIONARY_TIME - elapsed_time} seconds more"
            )

        # Update the timestamp each time before publishing
        joint_state_msg.header.stamp = rospy.Time.now()

        pub.publish(joint_state_msg)
        time_since_last_publish = (rospy.Time.now() - last_publish_time).to_sec()
        if time_since_last_publish > 0.2:
            rospy.loginfo("\n" + "=" * 80)
            rospy.loginfo("SLOW")
        rospy.loginfo(
            f"Publishing {np.round(time_since_last_publish * 1000)} ms since last publish, {np.round(1./time_since_last_publish)} Hz)"
        )
        if time_since_last_publish > 0.2:
            rospy.loginfo("SLOW")
            rospy.loginfo("\n" + "=" * 80 + "\n")
        last_publish_time = rospy.Time.now()
        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("iiwa_joint_publisher", anonymous=True)
        init_joint_pos = get_initial_joint_pos()
        publish_joint_cmd(init_joint_pos=init_joint_pos)
    except rospy.ROSInterruptException:
        pass
