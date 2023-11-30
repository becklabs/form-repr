import numpy as np
from collections import defaultdict

def parse_pose_json(json_dict):
    """
    Parse a MMPose json file into a list of numpy arrays, each representing a motion of a single subject. 
    """
    tracks = defaultdict(list)
    for frame in json_dict:
        for i, track in enumerate(frame["predictions_2d"]):
            keypoints = np.array(track["keypoints"])
            keypoint_scores = np.array(track["keypoint_scores"])
            tracks[i].append(np.column_stack((keypoints, keypoint_scores)))
    for i in tracks:
        tracks[i] = np.array(tracks[i])
    return list(tracks.values())  # (n_tracks, n_frames, n_keypoints, 3)

def non_overlapping_segments(array, length_threshold=30, score_threshold=10.2):
    """
    Given a 3D array of shape (n_frames, n_keypoints, 3), return a list of non-overlapping segments of the array.
    Each segment is a 3D array of shape (segment_length, n_keypoints, 3).
    """
    n_frames = array.shape[0]
    all_segments = []

    # Find all possible segments
    for start in range(n_frames):
        total_score = 0
        for end in range(start, n_frames):
            frame = array[end]
            frame_score = np.sum(frame[:, 2])
            total_score += frame_score

            segment_length = end - start + 1
            if segment_length >= length_threshold:
                avg_score = total_score / segment_length
                if avg_score >= score_threshold:
                    all_segments.append((start, end, segment_length))
                    break

    # Sort segments by their length in descending order
    all_segments.sort(key=lambda x: x[2], reverse=True)

    selected_segments = []
    used_frames = set()

    # Select non-overlapping, longest segments
    for segment in all_segments:
        start, end, _ = segment
        if not any(frame in used_frames for frame in range(start, end + 1)):
            selected_segments.append(array[start:end + 1])
            used_frames.update(range(start, end + 1))

    return selected_segments

def overlapping_segments(array, length_threshold=30, score_threshold=10.2):
    """
    Given a 3D array of shape (n_frames, n_keypoints, 3), return a list of possibly overlapping segments of the array.
    Each segment is a 3D array of shape (segment_length, n_keypoints, 3).
    """
    n_frames = array.shape[0]
    segments = []

    for start in range(n_frames):
        total_score = 0
        for end in range(start, n_frames):
            frame = array[end]
            frame_score = np.sum(frame[:, 2])
            total_score += frame_score

            segment_length = end - start + 1
            if segment_length >= length_threshold:
                avg_score = total_score / segment_length
                if avg_score >= score_threshold:
                    segments.append(array[start:end+1])
                    break  # Found a qualifying segment starting at 'start', move to next starting frame

    return segments