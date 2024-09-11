import os

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_data(joint_states_filename: str, joint_cmds_filename: str) -> None:
    """Plot joint states and commands and save the plots as files."""
    # Load the data from the .npy files
    joint_states = np.load(joint_states_filename)
    joint_cmds = np.load(joint_cmds_filename)

    # Ensure both files have the same number of samples and 7 joints
    assert (
        joint_states.shape == joint_cmds.shape
    ), "Joint states and commands must have the same shape"
    assert joint_states.shape[1] == 7, "Expected data for 7 joints"

    # Get the number of time steps (T)
    T = joint_states.shape[0]

    # Generate a time axis for plotting (assuming 60Hz sampling rate)
    time_axis = np.arange(T) / 60.0  # Time in seconds

    # Extract the base filename without extension
    base_filename = os.path.splitext(joint_states_filename)[0]

    # Create and save a plot for each joint
    for joint_idx in range(7):
        plt.figure()

        # Plot joint state and command for the current joint
        plt.plot(time_axis, joint_states[:, joint_idx], label="Joint State")
        plt.plot(
            time_axis, joint_cmds[:, joint_idx], label="Joint Command", linestyle="--"
        )

        # Add legend, title, and labels
        plt.legend()
        plt.title(f"Joint {joint_idx + 1} - Position")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")

        # Save the figure with a filename that includes the joint number
        plot_filename = f"{base_filename}_joint_{joint_idx + 1}.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"Saved plot: {plot_filename}")


if __name__ == "__main__":
    # Example usage
    joint_states_file = "20240910_153012_joint_states.npy"
    joint_cmds_file = "20240910_153012_joint_cmds.npy"

    plot_joint_data(joint_states_file, joint_cmds_file)
