## Setup Instructions

### Initialize Submodules

```bash
git submodule update --init --recursive
```

## MMPose Setup

For MPS (Metal Performance Shaders) support, install the nightly build of PyTorch and torchvision using the following command:

```bash
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

For regular installation without MPS support, install PyTorch using:

```bash
pip3 install torch
```

Install the wheel package:

```bash
pip3 install wheel
```

Update and install openmim:

```bash
pip3 install -U openmim
```

Install mmengine:

```bash
mim install mmengine
```

Install mmcv with CUDA ops:

```bash
cd mmcv && MMCV_WITH_OPS=1 pip3 install -e . && cd ..
```

Install mmdet (version 3.1.0 or later):

```bash
mim install "mmdet>=3.1.0"
```

Download MMPose checkpoints:

```bash
mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest checkpoints/mmpose
mim download mmpose --config motionbert_dstformer-ft-243frm_8xb32-120e_h36m --dest checkpoints/mmpose
```

## MotionBERT Setup


### Install Requirements

Navigate to the MotionBERT directory and install the required packages:

```bash
cd MotionBERT && pip3 install -r requirements.txt && cd ..
```

### Download Weights and Data

1. Download the latest MotionBERT-Lite weights from the [MotionBERT GitHub repository](https://github.com/Walter0807/MotionBERT).

   - Place the `latest_epoch.bin` file in the `checkpoints/pretrain/MB_release/` directory.

2. Download the preprocessed Human3.6m dataset from the [MotionBERT documentation](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md).

   - Place the `.pkl` file in the `data/motion3d/h3m_sh_conf_cam_souce_final.pkl` directory.

### Convert Human3.6m Data

Run the conversion script for the Human3.6m dataset:

```bash
cd tools && python3 convert_h36m.py && cd ..
```

## Scripts

### Pose Estimation

To extract keypoints from a video using `pose_estimation.py`, run the following command:

```sh
python tools/pose_estimation.py --video_path <path_to_video> --output_path <path_to_output_json> --is_3d <True_or_False> --overwrite <True_or_False>
```

- `--video_path`: Path to the input video file.
- `--output_path`: Path to save the extracted keypoints in JSON format.
- `--is_3d`: Set to `True` to extract 3D keypoints, `False` for 2D keypoints.
- `--overwrite`: Set to `True` to overwrite existing output files, `False` otherwise.

### Calculate Embeddings

To calculate embeddings for a 2D pose sequence using `get_embedding.py`, use the following command:

```sh
python tools/get_embedding.py --checkpoint <path_to_model_checkpoint> --input_path <path_to_json_files> --output_path <path_to_save_embeddings>
```

- `--checkpoint`: Path to the model checkpoint file.
- `--input_path`: Path to the folder containing JSON files with 2D pose sequences.
- `--output_path`: Path to save the resulting embeddings as `.npy` files.

### Fine-Tuning MotionBERT

To fine-tune MotionBERT on the running dataset using `train.py`, execute the following command:

```sh
python train.py --config <path_to_config_yaml> --checkpoint <path_to_checkpoint_dir> --pretrained <path_to_pretrained_checkpoint> 
```

- `--config`: Path to the YAML configuration file.
- `--checkpoint`: Directory to save checkpoints.
- `--pretrained`: Directory containing the pretrained checkpoint.
