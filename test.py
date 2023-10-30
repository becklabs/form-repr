import numpy as np

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'mmpose/tests/data/coco/000000000785.jpg'
)

keypoints = np.array(results[-1].pred_instances.keypoints)
print(keypoints.shape) # (1, 17, 2)

# Plot the keypoints ontop of the image
# First we need to load the image
import cv2
img = cv2.imread('mmpose/tests/data/coco/000000000785.jpg')

# Then we can plot the keypoints (just plot the xy coordinates as red dots onto the image)

for i in range(keypoints.shape[1]):
    cv2.circle(img, (int(keypoints[0,i,0]), int(keypoints[0,i,1])), 3, (0,0,255), -1)

# Finally we can show the image
cv2.imshow('image', img)

# Wait for the user to press a key
cv2.waitKey(0)


