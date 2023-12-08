import json
import numpy as np
from collections import defaultdict


def calculate_distance(person1_keypoints, person2_keypoints):
    """
    Calculate the Euclidean distance between two sets of keypoints.
    """
    return np.linalg.norm(np.array(person1_keypoints) - np.array(person2_keypoints), ord=2)


def track_people_across_frames(keypoints_2d, keypoints_scores):
    num_frames = len(keypoints_2d)
    tracks = defaultdict(list)

    for i in range(num_frames - 1):
        current_frame, current_scores = keypoints_2d[i], keypoints_scores[i]
        current_scores = np.mean(current_scores, axis=1)
        # Sort current frame by highest score first
        # current_frame = [x for _, x in sorted(zip(current_scores, current_frame), key=lambda pair: pair[0], reverse=True)]

        last_frame, last_scores = keypoints_2d[i-1], keypoints_scores[i-1]

        if i == 0:
            for j in range(len(current_frame)):
                tracks[j] = [j]  # Each person starts with their own track
        else :
            distance_matrix = np.zeros((len(current_frame), len(last_frame)))

            # Compute distance matrix
            for j, (person1, scores1) in enumerate(zip(current_frame, current_scores)):
                for k, (person2, scores2) in enumerate(zip(last_frame, last_scores)):
                    distance_matrix[j, k] = calculate_distance(person1, person2)
            

            # Match people in next frame
            for j, person1 in enumerate(current_frame):
                if distance_matrix.shape[1] == 0:
                    break
                
                matched_person_index = np.argmin(distance_matrix[j, :])
                distance_matrix = np.delete(distance_matrix, matched_person_index, 1)

                for num in tracks:
                    print(len(tracks[num]))
                    if tracks[num][i-1] == matched_person_index:
                        tracks[num].append(j)

            for num in tracks:
                if len(tracks[num]) < i+1:
                    tracks[num].append(tracks[num][-1])
    return tracks

def rearrange_keypoints(keypoints_2d, tracks):
    rearranged_keypoints = []

    for frame_index in range(len(keypoints_2d)):
        frame = keypoints_2d[frame_index]
        new_frame = [None] * len(tracks)  

        for person_index, track in tracks.items():
            if frame_index < len(track):
                tracked_person_index = track[frame_index]
                if tracked_person_index < len(frame):
                    new_frame[person_index] = frame[tracked_person_index]

        new_frame = [person for person in new_frame if person is not None]
        rearranged_keypoints.append(new_frame)

    return rearranged_keypoints

if __name__ == "__main__":
    with open('../data/poses/Pro_Runners/Donavan_Brazier.json') as f:
        frames = json.load(f)

        keypoints_2d = [[f['keypoints'] for f in frame['predictions_2d']] for frame in frames]
        keypoint_scores = [[f['keypoint_scores'] for f in frame['predictions_2d']] for frame in frames]







