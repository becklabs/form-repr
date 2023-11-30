from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
import random
import sys
from .utils import parse_pose_json, non_overlapping_segments

current_dir = os.path.dirname(__file__)
submodule_lib_path = os.path.join(current_dir, "../../MotionBERT")
sys.path.append(submodule_lib_path)

from MotionBERT.lib.utils.utils_data import crop_scale, flip_data


class PoseTrackDataset2D(Dataset):
    def __init__(self, data_root, flip=True, scale_range=[0.25, 1]):
        super(PoseTrackDataset2D, self).__init__()
        self.flip = flip
        file_list = sorted(os.listdir(data_root))
        all_motions = []
        all_motions_filtered = []
        self.scale_range = scale_range
        for filename in file_list:
            with open(os.path.join(data_root, filename), "r") as file:
                json_dict = json.load(file)
                motions = parse_pose_json(json_dict)
            for motion in motions:
                all_motions += non_overlapping_segments(motion, length_threshold=30)
        for motion in all_motions:
            if len(motion) < 30:
                continue
            # Split motion into segments of 30 frames
            for i in range(len(motion) // 30):
                motion_segment = motion[i * 30 : (i + 1) * 30] 
                if (
                    np.sum(motion_segment[:, :, 2]) / motion_segment.shape[0] <= 10.2
                ):  # Valid joint num threshold
                    continue
                motion_segment = crop_scale(motion_segment, self.scale_range)
                # motion_segment[motion_segment[:, :, 2] == 0] = 0
                if not np.all(motion_segment[:, 0, 2] > 0.5):
                    continue  # Root all visible (needed for framewise rootrel)
                all_motions_filtered.append(motion_segment)
        all_motions_filtered = np.array(all_motions_filtered)
        self.motions_2d = all_motions_filtered

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.motions_2d)

    def __getitem__(self, index):
        "Generates one sample of data"
        motion_2d = torch.FloatTensor(self.motions_2d[index])
        if self.flip and random.random() > 0.5:
            motion_2d = flip_data(motion_2d)
        return motion_2d, motion_2d
