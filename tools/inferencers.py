import os
import torch
import inspect
from mmpose.apis.inferencers import Pose2DInferencer, Pose3DInferencer

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), '..')


# Equivalent to MMPoseInferencer(pose2d='human', device='cpu')
inferencer_2d = Pose2DInferencer(
    model=os.path.join(
        ROOT_DIR,
        "mmpose/configs/body_2d_keypoint/rtmpose/coco/"
        "rtmpose-m_8xb256-420e_coco-256x192.py",
    ),
    weights=os.path.join(
        ROOT_DIR,
        "checkpoints/mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
    ),
    device=device,
)

# Equivalent to MMPoseInferencer(pose3d='human3d', device='cpu')
inferencer_3d = Pose3DInferencer(
    pose2d_model=os.path.join(
        ROOT_DIR,
        "mmpose/configs/body_2d_keypoint/rtmpose/coco/"
        "rtmpose-m_8xb256-420e_coco-256x192.py",
    ),
    pose2d_weights=os.path.join(
        ROOT_DIR,
        "checkpoints/mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
    ),
    model=os.path.join(ROOT_DIR, "mmpose/configs/body_3d_keypoint/motionbert/h36m/",
    "motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"),
    weights=os.path.join(
        ROOT_DIR, "checkpoints/mmpose/motionbert_ft_h36m-d80af323_20230531.pth"
    ),
    device=device,
)


inferencer_3d_basic = Pose3DInferencer(
    model=os.path.join(ROOT_DIR, "mmpose/configs/body_3d_keypoint/motionbert/h36m/",
    "motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"),
    weights=os.path.join(
        ROOT_DIR, "checkpoints/mmpose/motionbert_ft_h36m-d80af323_20230531.pth"
    ),
    device=device,
)
