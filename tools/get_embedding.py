
import os
import sys
import argparse

current_dir = os.path.dirname(__file__)
submodule_lib_path = os.path.join(current_dir, "../MotionBERT")
src_lib_module = os.path.join(current_dir, "../src")
sys.path.append(submodule_lib_path)
sys.path.append(src_lib_module)
from tqdm import tqdm
from torch.utils.data import DataLoader


from MotionBERT.lib.utils.learning import *
from MotionBERT.lib.model.DSTformer import DSTformer
from MotionBERT.lib.utils.utils_data import flip_data
from src.data.dataset_wild import WildDetDataset


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    model_backbone = DSTformer(maxlen=  243, dim_feat= 256, mlp_ratio= 4,depth= 5,dim_rep= 512,num_heads= 8,att_fuse= True)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
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
    for file_name in [file for file in os.listdir(args.input_path) if file.endswith('.json')]:
        wild_dataset = WildDetDataset(args.input_path + file_name, clip_len=243, scale_range=[1, 1], focus=None)

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
        np.save(args.output_path + file_name.replace('.json', '') + '.npy', results_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate embedding for a 2d pose sequence")
    parser.add_argument("--checkpoint", type=str, default='../checkpoints/pose3d/latest_epoch.bin',  help="path the model checkpoint")
    parser.add_argument("--input_path", type=str, default='../data/poses/oliver/', help="path to the folder containing the json 2d pose sequence files")
    parser.add_argument("--output_path", type=str, default='../data/embed/oliver/', help="path to save the embeddings")
    args = parser.parse_args()
    print(args)
    main(args)








