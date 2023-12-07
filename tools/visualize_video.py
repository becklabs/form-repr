import cv2
import numpy as np
import json
import os

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
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Write the frame
        out.write(frame)

        frame_idx += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
import cv2
import numpy as np

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

# Example usage
# keypoints = np.random.rand(100, 20, 2) # Example keypoints
# draw_keypoints_on_video(keypoints, 'input_video.mp4', 'output_video.mp4')

vname = 'IMG_5040'
with open(f'../data/poses/oliver/{vname}.json') as f:
    frames = json.load(f)

keypoints_2d = np.array([frame['predictions_2d'][0]['keypoints'] for frame in frames])
print(keypoints_2d.shape)

# keypoints = np.random.rand(100, 20, 2) # Example keypoints
draw_keypoints_on_video(keypoints_2d, f'../data/video/oliver/{vname}.mp4', 'overstride.mp4')
