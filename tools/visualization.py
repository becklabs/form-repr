import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pose_estimation import extract_video_keypoints
from fix_tracks import track_people_across_frames, rearrange_keypoints
from inferencers import inferencer_2d_yolo


def visualize_video_3d(keypoints, keypoint_scores, keypoint_threshold=0.3):
    skeleton_pairs = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
        [7, 11], [11, 12], [12, 13], [7, 14], [14, 15], [15, 16]
    ]

    all_keypoints = np.array(keypoints)
    max_range = np.ptp(all_keypoints, axis=(0, 1)).max() / 2.0
    mid_x = np.mean(all_keypoints[:, :, 0])
    mid_y = np.mean(all_keypoints[:, :, 1])
    mid_z = np.mean(all_keypoints[:, :, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter_plot = ax.scatter([], [], [])
    lines = [ax.plot([], [], [])[0] for _ in skeleton_pairs]

    def update(frame_ind):
        ax.clear()  # Clear the current axes
        keypoints_frame = keypoints[frame_ind]
        scores = keypoint_scores[frame_ind]

        filtered_keypoints = [kp if scores[i] > keypoint_threshold else (np.nan, np.nan, np.nan) for i, kp in
                              enumerate(keypoints_frame)]
        x, y, z = zip(*filtered_keypoints)

        ax.scatter(x, y, z, c='blue')

        for pair in skeleton_pairs:
            xs = [filtered_keypoints[pair[0]][0], filtered_keypoints[pair[1]][0]]
            ys = [filtered_keypoints[pair[0]][1], filtered_keypoints[pair[1]][1]]
            zs = [filtered_keypoints[pair[0]][2], filtered_keypoints[pair[1]][2]]
            ax.plot(xs, ys, zs, c='red')

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_box_aspect([1, 1, 1])  

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return [scatter_plot] + lines

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)

    plt.show()


def visualize_video_2d(keypoints, keypoint_scores, keypoint_threshold=0.3):
    all_keypoints = np.array([kp for frame in keypoints for kp in frame if kp is not None])
    max_range = np.ptp(all_keypoints, axis=0).max() / 2.0
    mid_x = np.mean(all_keypoints[:, 0])
    mid_y = np.mean(all_keypoints[:, 1])

    fig, ax = plt.subplots()

    scatter_plot = ax.scatter([], [])

    # Animation update function
    def update(frame_ind):
        ax.clear()  
        keypoints_frame = keypoints[frame_ind]
        scores = keypoint_scores[frame_ind]

        # Filter keypoints based on the score
        filtered_keypoints = [kp if scores[i] > keypoint_threshold else (np.nan, np.nan) for i, kp in
                              enumerate(keypoints_frame)]
        x, y = zip(*filtered_keypoints)

        x = np.array(x)
        y = np.array(y)
        print(y)

        ax.scatter(x, y, c='blue')

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim((mid_y - max_range), (mid_y + max_range))

        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.invert_xaxis()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')


        return [scatter_plot] # + lines

    ani = FuncAnimation(fig, update, frames=len(keypoints), blit=False)

    plt.show()

def visualize_multiple_2d(keypoints, keypoint_scores, keypoint_threshold=0.3):
    scatterplot_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'] * 10

    all_keypoints = []
    for f in keypoints:
        for t in f:
            for k in t:
                if k is not None:
                    all_keypoints.append(k)
    all_keypoints = np.array(all_keypoints)
    print(all_keypoints.shape)

    max_range = np.ptp(all_keypoints, axis=0).max() / 2.0
    mid_x = np.mean(all_keypoints[:, 0])
    mid_y = np.mean(all_keypoints[:, 1])

    print(max_range, mid_x, mid_y)

    # Create 2D plot
    fig, ax = plt.subplots()

    scatter_plot = ax.scatter([], [])
    # lines = [ax.plot([], [])[0] for _ in skeleton_pairs]

    def update(frame_ind):
        ax.clear()  
        keypoints_frame = keypoints[frame_ind]
        scores = keypoint_scores[frame_ind]

        for i, (track_keypoints, track_scores) in enumerate(zip(keypoints_frame, scores)):
            # Filter keypoints based on the score
            filtered_keypoints = [kp if track_scores[i] > keypoint_threshold else (np.nan, np.nan) for i, kp in enumerate(track_keypoints)]
            x, y = zip(*filtered_keypoints)

            x = np.array(x)
            y = np.array(y)

            # Redraw scatter plot for filtered keypoints
            ax.scatter(x, y, c=scatterplot_colors[i])
            ax.legend(['Track {}'.format(i) for i in range(len(keypoints_frame))])

        # Use fixed axis limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim((mid_y - max_range), (mid_y + max_range))


        ax.invert_yaxis()
        # Set aspect ratio to be equal
        ax.set_aspect('equal', adjustable='box')

        # Set axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        return [scatter_plot] # + lines

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(keypoints), blit=False)

    plt.show()

def draw_keypoints_on_video(keypoints, video_path, output_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if there are keypoints for this frame
        if frame_idx < len(keypoints):
            for x, y in keypoints[frame_idx]:
                # Draw the keypoints
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Write the frame
        out.write(frame)

        frame_idx += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # with open('../data/poses/All_Data/0.json') as f:
    #     frames = json.load(f)

    # keypoints_3d = [frame['predictions_3d'][0]['keypoints'] for frame in frames]

    # keypoints_2d = [frame['predictions_2d'][0]['keypoints'] for frame in frames]
    # keypoint_scores = [frame['predictions_2d'][0]['keypoint_scores'] for frame in frames]

    # Apply the function to your data
    # tracks = track_people_across_frames(keypoints_2d, keypoint_scores)

    # person_0 = tracks[0]
    # person_0_keypoints = []
    # person_0_keypoint_scores = []
    # for frame, scores, ind in zip(keypoints_2d, keypoint_scores, person_0):
    #     if ind is not None:
    #         person_0_keypoints.append([frame[ind]])
    #         person_0_keypoint_scores.append([scores[ind]])
        

    # # Rearrange the keypoints according to the tracks
    # rearranged_keypoints = rearrange_keypoints(keypoints_2d, tracks)
    # rearranged_keypoint_scores = rearrange_keypoints(keypoint_scores, tracks)

    # visualize_video_3d(keypoints_3d, keypoint_scores, keypoint_threshold=0.3)

    # frames = extract_video_keypoints('../data/video/Pro_Runners/Katelyn_Tuohy.mp4', inferencer_2d_yolo, is_3d=False)
    # keypoints_2d = [[f['keypoints'] for f in frame['predictions_2d']] for frame in frames]
    # keypoint_scores = [[f['keypoint_scores'] for f in frame['predictions_2d']] for frame in frames]
    # visualize_video_2d(keypoints_2d, keypoint_scores, keypoint_threshold=0.3)

    # vname = 'IMG_5040'
    # with open(f'../data/poses/oliver/{vname}.json') as f:
    #     frames = json.load(f)

    # keypoints_2d = np.array([frame['predictions_2d'][0]['keypoints'] for frame in frames])
    # print(keypoints_2d.shape)

    # # keypoints = np.random.rand(100, 20, 2) # Example keypoints
    # draw_keypoints_on_video(keypoints_2d, f'../data/video/oliver/{vname}.mp4', 'overstride.mp4')
    pass