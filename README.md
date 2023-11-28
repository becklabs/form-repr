#### MMPose Setup
For mps:
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu

Regular:
pip3 install torch

pip3 install wheel

pip3 install -U openmim

mim install mmengine

cd mmcv
$MMCV_WITH_OPS=1 pip3 install -e .

cd ..

mim install "mmdet>=3.1.0"

mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192  --dest checkpoints/mmpose 
mim download mmpose --config motionbert_dstformer-ft-243frm_8xb32-120e_h36m --dest checkpoints/mmpose



