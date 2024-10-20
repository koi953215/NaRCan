import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from models.model import Homography, Siren
from utils.util import apply_homography, TestVideoFitting, get_mgrid, read_specific_lines_in_order
from utils.linear_interpolation import linear_interpolate


def reconstruct_frames(name, sep, g_old, data_path, output_path, save_canonical=False):
    pth_path = os.path.join(output_path, "pth_file")
    
    f_path = os.path.join(pth_path, f"mlp_f{sep}.pth")
    f = Siren(in_features=2, out_features=3, hidden_features=256, 
            hidden_layers=5, outermost_linear=True)
    f.load_state_dict(torch.load(f_path))
    f.cuda()
    f.eval()
    
    g_path = os.path.join(pth_path, f"mlp_g{sep}.pth")
    g = Siren(in_features=3, out_features=2, hidden_features=256,
              hidden_layers=5, outermost_linear=True)
    g.load_state_dict(torch.load(g_path))
    g.cuda()
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
            o = f(xy_)

            if step == 0:
                myoutput = o
            else:
                myoutput = torch.cat([myoutput, o])
                
        if save_canonical:
            # Save Canonical Images
            canonical_path = os.path.join(output_path, "original_canonical")
            os.makedirs(canonical_path, exist_ok=True)
            
            scale_factor = read_specific_lines_in_order(
                os.path.join(output_path, f"canonical_region.txt"))
            scale_factor = list(map(float, scale_factor))
            with torch.no_grad():
                xy = get_mgrid([512, 1024], [-scale_factor[0], -scale_factor[1]], [scale_factor[0], scale_factor[1]]).cuda()
                output = f(xy)
                output = output.view(512, 1024, 3).cpu().detach().numpy()
                output = np.clip(output, -1, 1) * 0.5 + 0.5
                output = Image.fromarray(np.uint8(output * 255))
                output.save(os.path.join(canonical_path, f"canonical_{sep}.png"))

    # Reconstruction
    reconstruction_path = os.path.join(output_path, "reconstruction", f"{name}_{sep}")
    os.makedirs(reconstruction_path, exist_ok=True)
    myoutput = myoutput.reshape(v.H, v.W, data_len, 3).permute(2, 0, 1, 3).clone().detach().cpu().numpy().astype(np.float32)
    myoutput = np.clip(myoutput, -1, 1) * 0.5 + 0.5

    filenames = sorted(os.listdir(now_path))
    for k in range(len(myoutput)):
        img = Image.fromarray(np.uint8(myoutput[k] * 255)).resize((v.W, v.H))
        img.save(os.path.join(reconstruction_path, filenames[k]))
        
  
def test(scene_name, separate_num, save_canonical):
    data_path = os.path.join("data", scene_name, f"separate_{separate_num}")
    output_path = os.path.join("output", scene_name, f"separate_{separate_num}")
    
    g_old_path = os.path.join(output_path, "pth_file", "homography_g.pth")
    g_old = Homography(hidden_features=256, hidden_layers=2).cuda()
    g_old.load_state_dict(torch.load(g_old_path))
    g_old.eval()
    
    for sep in range(1, separate_num + 1):        
        reconstruct_frames(scene_name, sep, g_old, data_path, output_path, save_canonical)
        
    reconstruction_path = os.path.join(output_path, "reconstruction")
    linear_interpolate(scene_name, reconstruction_path, reconstruction_path, separate_num, save_video=True)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help='scene_name')
    parser.add_argument('--separate_num', '-sn', type=int, default=3, help='Number of separations.')
    parser.add_argument('--save_canonical', action="store_true")
    
    args = parser.parse_args()
    
    test(args.name, args.separate_num, args.save_canonical)