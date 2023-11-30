import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from pose_estimation import extract_video_keypoints
# from inferencers import inferencer_2d

with open('../data/poses/All_Data/0.json') as f:
    frames = json.load(f)

def visualize_video_3d(keypoints, keypoint_scores, keypoint_threshold=0.3):
    skeleton_pairs = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
        [7, 11], [11, 12], [12, 13], [7, 14], [14, 15], [15, 16]
    ]

    # Calculate global axis limits
    all_keypoints = np.array(keypoints)
    max_range = np.ptp(all_keypoints, axis=(0, 1)).max() / 2.0
    mid_x = np.mean(all_keypoints[:, :, 0])
    mid_y = np.mean(all_keypoints[:, :, 1])
    mid_z = np.mean(all_keypoints[:, :, 2])

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize scatter plot and lines for skeleton
    scatter_plot = ax.scatter([], [], [])
    lines = [ax.plot([], [], [])[0] for _ in skeleton_pairs]

    # Animation update function
    def update(frame_ind):
        ax.clear()  # Clear the current axes
        keypoints_frame = keypoints[frame_ind]
        scores = keypoint_scores[frame_ind]

        # Filter keypoints based on the score
        filtered_keypoints = [kp if scores[i] > keypoint_threshold else (np.nan, np.nan, np.nan) for i, kp in enumerate(keypoints_frame)]
        x, y, z = zip(*filtered_keypoints)

        # Redraw scatter plot for filtered keypoints
        ax.scatter(x, y, z, c='blue')

        # Redraw skeleton lines
        for pair in skeleton_pairs:
            xs = [filtered_keypoints[pair[0]][0], filtered_keypoints[pair[1]][0]]
            ys = [filtered_keypoints[pair[0]][1], filtered_keypoints[pair[1]][1]]
            zs = [filtered_keypoints[pair[0]][2], filtered_keypoints[pair[1]][2]]
            ax.plot(xs, ys, zs, c='red')

        # Use fixed axis limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set aspect ratio to be equal
        ax.set_box_aspect([1,1,1])  # Newer matplotlib versions

        # Set axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return [scatter_plot] + lines

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)

    plt.show()


def visualize_video_2d(keypoints, keypoint_scores, keypoint_threshold=0.3):
    skeleton_pairs = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
        [7, 11], [11, 12], [12, 13], [7, 14], [14, 15], [15, 16]
    ]

    # Calculate global axis limits
    all_keypoints = np.array([kp for frame in keypoints for kp in frame if kp is not None])
    max_range = np.ptp(all_keypoints, axis=0).max() / 2.0
    mid_x = np.mean(all_keypoints[:, 0])
    mid_y = np.mean(all_keypoints[:, 1])

    # Create 2D plot
    fig, ax = plt.subplots()

    # Initialize scatter plot and lines for skeleton
    scatter_plot = ax.scatter([], [])
    lines = [ax.plot([], [])[0] for _ in skeleton_pairs]

    # Animation update function
    def update(frame_ind):
        ax.clear()  # Clear the current axes
        keypoints_frame = keypoints[frame_ind]
        scores = keypoint_scores[frame_ind]

        # Filter keypoints based on the score
        filtered_keypoints = [kp if scores[i] > keypoint_threshold else (np.nan, np.nan) for i, kp in enumerate(keypoints_frame)]
        x, y = zip(*filtered_keypoints)

        # Redraw scatter plot for filtered keypoints
        ax.scatter(x, y, c='blue')

        # Redraw skeleton lines
        # for pair in skeleton_pairs:
        #     xs = [filtered_keypoints[pair[0]][0], filtered_keypoints[pair[1]][0]]
        #     ys = [filtered_keypoints[pair[0]][1], filtered_keypoints[pair[1]][1]]
        #     ax.plot(xs, ys, c='red')

        # Use fixed axis limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)

        # Set aspect ratio to be equal
        ax.set_aspect('equal', adjustable='box')

        # Set axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        return [scatter_plot] + lines

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(keypoints), blit=False)

    plt.show()


keypoints_3d = [frame['predictions_3d'][0]['keypoints'] for frame in frames]
# keypoints_2d = [frame['predictions_3d'][0]['keypoints'] for frame in frames]
keypoint_scores = [frame['predictions_2d'][0]['keypoint_scores'] for frame in frames]

visualize_video_3d(keypoints_3d, keypoint_scores, keypoint_threshold=0.3)