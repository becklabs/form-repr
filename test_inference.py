from mmpose.apis import MMPoseInferencer

# img_path = 'mmpose/tests/data/coco/000000000785.jpg'   # replace this with your own image path
img_path = 'test3.mp4'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human', device='cpu')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
results = [r for r in result_generator]