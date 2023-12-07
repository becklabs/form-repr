import json
import numpy as np
from collections import defaultdict

with open('../data/poses/Pro_Runners/Donavan_Brazier.json') as f:
    frames = json.load(f)

keypoints_2d = [[f['keypoints'] for f in frame['predictions_2d']] for frame in frames]
keypoint_scores = [[f['keypoint_scores'] for f in frame['predictions_2d']] for frame in frames]

# def calculate_distance(person1_keypoints, person2_keypoints, person1_scores, person2_scores, score_threshold=0.3):
#     """
#     Calculate the Euclidean distance between two sets of keypoints, considering only keypoints with a score above the threshold.
#     """
#     distance = 0
#     count = 0
#     for k1, k2, s1, s2 in zip(person1_keypoints, person2_keypoints, person1_scores, person2_scores):
#         if s1 >= score_threshold and s2 >= score_threshold:
#             distance += np.linalg.norm(np.array(k1) - np.array(k2))
#             count += 1
#     return distance if count > 0 else float('inf')

def calculate_distance(person1_keypoints, person2_keypoints):
    """
    Calculate the Euclidean distance between two sets of keypoints.
    """
    return np.linalg.norm(np.array(person1_keypoints) - np.array(person2_keypoints), ord=2)

# def track_across_frames(keypoints_2d):
#     num_frames = len(keypoints_2d)
#     tracks = defaultdict(list)

#     first_frame = keypoints_2d[0]

#     for i in range(num_frames - 1):
#         current_frame = keypoints_2d[i]
#         next_frame = keypoints_2d[i + 1]

#         distance_matrix = np.zeros((len(current_frame), len(next_frame)))
#         for j, (person1) in enumerate(current_frame):
#             for k, (person2) in enumerate(next_frame):
#                 distance_matrix[j, k] = calculate_distance(person1, person2)


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
            # Initialize tracks with first frame
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


# def track_people_across_frames(keypoints_2d, keypoints_scores, distance_threshold=30):
#     num_frames = len(keypoints_2d)
#     tracks = defaultdict(list)

#     for i in range(num_frames - 1):
#         current_frame, current_scores = keypoints_2d[i], keypoints_scores[i]
#         next_frame, next_scores = keypoints_2d[i + 1], keypoints_scores[i + 1]

#         if i == 0:
#             # Initialize tracks with first frame
#             for j in range(len(current_frame)):
#                 tracks[j].append(j)  # Each person starts with their own track

#         distance_matrix = np.full((len(current_frame), len(next_frame)), np.inf)

#         # Compute distance matrix
#         for j, (person1, scores1) in enumerate(zip(current_frame, current_scores)):
#             for k, (person2, scores2) in enumerate(zip(next_frame, next_scores)):
#                 distance = calculate_distance(person1, person2)
#                 if distance < distance_threshold:
#                     distance_matrix[j, k] = distance

#         # Match people in next frame
#         for j, person1 in enumerate(current_frame):
#             matched_person_index = np.argmin(distance_matrix[j, :])
#             min_distance = distance_matrix[j, matched_person_index]

#             if min_distance < distance_threshold:
#                 tracks[j].append(matched_person_index)
#             else:
#                 tracks[j].append(None)  # No match found within the threshold

#     # Handle any new people in the last frame
#     last_frame_index = len(keypoints_2d) - 1
#     for k in range(len(keypoints_2d[last_frame_index])):
#         if not any(k in track for track in tracks.values()):
#             new_track_id = max(tracks.keys()) + 1
#             tracks[new_track_id] = [None] * last_frame_index + [k]

#     return tracks

# print(track_people_across_frames(keypoints_2d, keypoint_scores))

def rearrange_keypoints(keypoints_2d, tracks):
    rearranged_keypoints = []

    for frame_index in range(len(keypoints_2d)):
        frame = keypoints_2d[frame_index]
        new_frame = [None] * len(tracks)  # Initialize based on the number of tracks

        for person_index, track in tracks.items():
            if frame_index < len(track):
                tracked_person_index = track[frame_index]
                # Check if the person is present in this frame
                if tracked_person_index < len(frame):
                    new_frame[person_index] = frame[tracked_person_index]

        # Remove None entries for any missing person in the frame
        new_frame = [person for person in new_frame if person is not None]
        rearranged_keypoints.append(new_frame)

    return rearranged_keypoints

# def rearrange_keypoints(keypoints_2d, tracks):
#     rearranged_keypoints = []
#     max_people = max(len(frame) for frame in keypoints_2d)  # Find the maximum number of people in any frame

#     for frame_index in range(len(keypoints_2d)):
#         frame = keypoints_2d[frame_index]
#         new_frame = [None] * max_people  # Initialize with the maximum number of people

#         for person_index in tracks:
#             track = tracks[person_index]
#             if frame_index < len(track):
#                 rearranged_index = track[frame_index]
#                 if rearranged_index < len(frame):
#                     new_frame[person_index] = frame[rearranged_index]

#         # Filter out None values if the person is not present in the frame
#         new_frame = [person for person in new_frame if person is not None]
        
#         rearranged_keypoints.append(new_frame)

#     return rearranged_keypoints


# Apply the function to your data
# tracks = track_people_across_frames(keypoints_2d, keypoint_scores)

# # Rearrange the keypoints according to the tracks
# rearranged_keypoints = rearrange_keypoints(keypoints_2d, tracks)

# print(rearranged_keypoints)

# def calculate_distance(person1_keypoints, person2_keypoints):
#     """
#     Calculate the Euclidean distance between two sets of keypoints.
#     """
#     return np.linalg.norm(np.array(person1_keypoints) - np.array(person2_keypoints))

# def track_people_across_frames(keypoints_2d):
#     num_frames = len(keypoints_2d)
#     tracks = defaultdict(list)

#     for i in range(num_frames - 1):
#         current_frame = keypoints_2d[i]
#         next_frame = keypoints_2d[i + 1]

#         if i == 0:
#             # Initialize tracks with first frame
#             for j in range(len(current_frame)):
#                 tracks[j] = [j]  # Each person starts with their own track

#         distance_matrix = np.zeros((len(current_frame), len(next_frame)))

#         # Compute distance matrix
#         for j, person1 in enumerate(current_frame):
#             for k, person2 in enumerate(next_frame):
#                 distance_matrix[j, k] = calculate_distance(person1, person2)

#         # Match people in next frame
#         for j, person1 in enumerate(current_frame):
#             matched_person_index = np.argmin(distance_matrix[j, :])
#             tracks[j].append(matched_person_index)

#     return tracks








