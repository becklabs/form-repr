from mmpose.apis import MMPoseInferencer

img_path = 'mmpose/tests/data/coco/000000000785.jpg'   # replace this with your own image path
img_path = 'test_videos/images.jpg'   # replace this with your own image path
img_path = 'test_videos/44.mp4'   # replace this with your own image path
# img_path = 'test_videos/test3.mp4'   # replace this with your own image path

inferencer = MMPoseInferencer('human', device='cpu')
# inferencer = MMPoseInferencer(pose2d='rtmpose-s_8xb1024-700e_body8-halpe26-256x192',
#                               pose2d_weights='rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth',
#                               device='cpu')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
results = [r for r in result_generator]
