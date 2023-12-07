
import os

import sys

current_dir = os.path.dirname(__file__)
submodule_lib_path = os.path.join(current_dir, "./MotionBERT")
sys.path.append(submodule_lib_path)
from tqdm import tqdm
from torch.utils.data import DataLoader


from MotionBERT.lib.utils.learning import *
from MotionBERT.lib.model.DSTformer import DSTformer
from MotionBERT.lib.utils.utils_data import flip_data
from src.data.dataset_wild import WildDetDataset

CHECKPOINT_PATH = 'checkpoints/pose3d/latest_epoch.bin'
INPUT_PATH = 'data/poses/oliver/'
OUTPUT_PATH = 'data/embed/oliver/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


model_backbone = DSTformer(maxlen=  243, dim_feat= 256, mlp_ratio= 4,depth= 5,dim_rep= 512,num_heads= 8,att_fuse= True)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=lambda storage, loc: storage)
if not torch.cuda.is_available():
  checkpoint['model_pos'] = {key.replace("module.", ""): value for key, value in checkpoint['model_pos'].items()}
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()

testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last': False
}

def main():
  for file_name in [file for file in os.listdir(INPUT_PATH) if file.endswith('.json')]:
      wild_dataset = WildDetDataset(INPUT_PATH + file_name, clip_len=243, scale_range=[1, 1], focus=None)

      test_loader = DataLoader(wild_dataset, **testloader_params)

      results_all = []
      with torch.no_grad():
          for batch_input in tqdm(test_loader):
              N, T = batch_input.shape[:2]
              if torch.cuda.is_available():
                  batch_input = batch_input.cuda()
              batch_input = batch_input[:, :, :, :3]

              batch_input_flip = flip_data(batch_input)
              predicted_3d_pos_1 = model_pos.get_representation(batch_input)
              predicted_3d_pos_flip = model_pos.get_representation(batch_input_flip)
              predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
              predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0

              results_all.append(predicted_3d_pos.cpu().numpy())

      results_all = np.hstack(results_all)
      results_all = np.concatenate(results_all)
      print(results_all)
      np.save(OUTPUT_PATH + file_name.replace('.json', '') + '.npy', results_all)








# Create Linear SVM object
# support = svm.LinearSVC(random_state=20)
# # Train the model using the training sets and check score on test dataset
#
#
# support.fit(x_train, y)
# predicted = support.predict(x_train)
# score=accuracy_score(y,predicted)
# print(score)

if __name__ == "__main__":
  main()