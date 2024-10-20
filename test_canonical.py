import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from models.model import Homography, Siren
from utils.util import apply_homography, TestVideoFitting, read_specific_lines_in_order
from utils.linear_interpolation import linear_interpolate


def reconstruct_using_canonical(name, data_path, output_path, separate_num):
    scale_idx = 0
    checkpoint_g_old = torch.load(os.path.join(output_path, "pth_file", "homography_g.pth"))
    g_old = Homography(hidden_features=256, hidden_layers=2).cuda()
    g_old.load_state_dict(checkpoint_g_old)
    g_old.eval()
    
    for sep in range(1, separate_num+1):
        scale_factor = read_specific_lines_in_order(
            os.path.join(output_path, f"canonical_region.txt"))
        scale_factor = list(map(float, scale_factor))

        checkpoint_g = torch.load(os.path.join(output_path, "pth_file", f"mlp_g{sep}.pth"))
        g = Siren(in_features=3, out_features=2, hidden_features=256,
            hidden_layers=5, outermost_linear=True).cuda()
        g.load_state_dict(checkpoint_g)
        g.eval()
        
        print("---Loading successfully---")
        now_path = os.path.join(data_path, f"{name}_{sep}")
        transform = Compose([
            Resize(512),
            ToTensor(),
            Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
        ])
        v = TestVideoFitting(now_path, transform)
        videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)

        model_input, ground_truth = next(iter(videoloader))
        model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

        myoutput = None
        data_len = len(os.listdir(now_path))

        with torch.no_grad():
            batch_size = (v.H * v.W)
            for step in range(data_len):
                start = (step * batch_size) % len(model_input)
                end = min(start + batch_size, len(model_input))

                # get the deformation
                xy, t = model_input[start:end, :-1], model_input[start:end, [-1]]
                xyt = model_input[start:end]
                h_old = apply_homography(xy, g_old(t))
                h = g(xyt)
                xy_ = h_old + h

                # use canonical to reconstruct
                w, h = v.W, v.H
                canonical_img_path = os.path.join(output_path, "edited_canonical", f"canonical_{sep}.png")
                canonical_img = np.array(Image.open(canonical_img_path).convert('RGB'))
                canonical_img = torch.from_numpy(canonical_img).float().cuda()
                h_c, w_c = canonical_img.shape[:2]
                grid_new = xy_.clone()
                
                grid_new[..., 1] = xy_[..., 0] / scale_factor[0]
                grid_new[..., 0] = xy_[..., 1] / scale_factor[1]
                # print(scale_idx + (i - 1), scale_idx + (i))

                if len(canonical_img.shape) == 3:
                    canonical_img = canonical_img.unsqueeze(0)
                
                results = torch.nn.functional.grid_sample(
                    canonical_img.permute(0, 3, 1, 2),
                    grid_new.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    padding_mode='border')
                o = results.squeeze().permute(1,0)

                if step == 0:
                    myoutput = o
                else:
                    myoutput = torch.cat([myoutput, o])
            
            myoutput = myoutput.reshape(v.H, v.W, data_len, 3).permute(2, 0, 1, 3).clone().detach().cpu().numpy().astype(np.float32)
            # myoutput = np.clip(myoutput, -1, 1) * 0.5 + 0.5

            edited_result_path = os.path.join(output_path, "edited_result", f"{name}_{sep}")
            os.makedirs(edited_result_path, exist_ok=True)
            
            filenames = sorted(os.listdir(now_path))
            for k in range(len(myoutput)):
                img = Image.fromarray(np.uint8(myoutput[k]))
                img.save(os.path.join(edited_result_path, filenames[k]))

            scale_idx += 1
        
  
def test(name, separate_num):
    data_path = os.path.join("data", name, f"separate_{separate_num}")
    output_path = os.path.join("output", name, f"separate_{separate_num}")
    
    reconstruct_using_canonical(name, data_path, output_path, separate_num)
    edited_result_path = os.path.join(output_path, "edited_result")
    linear_interpolate(name, edited_result_path, edited_result_path, separate_num, save_video=True)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create separation.")
    parser.add_argument('--name', '-n', type=str, required=True, help='scene_name')
    parser.add_argument('--separate_num', '-sn', type=int, default=3, help='Number of separations.')
    
    args = parser.parse_args()
    
    test(args.name, args.separate_num)