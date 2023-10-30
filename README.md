#### OpenPose Setup

git submodule update --init --recursive --remote
brew install cmake
brew install boost
brew install protobuf
brew install glog
brew install opencv

mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON -DGPU_MODE=CPU_ONLY

#### MMPose Setup
For mps:
pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu

Regular:
pip3 install torch

pip3 install wheel

pip3 install -U openmim

mim install mmengine

cd mmcv
$MMCV_WITH_OPS=1 pip3 install -e .

mim install "mmdet>=3.1.0"

mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .



