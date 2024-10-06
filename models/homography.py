import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from itertools import chain
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from model import Siren, Homography
from util import get_mgrid, apply_homography, jacobian, VideoFitting


def train_homography(path, total_steps, verbose=True, steps_til_summary=100):
    transform = Compose([
        Resize(512),
        ToTensor(),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])
    v = VideoFitting(path, transform, True)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)

    g = Homography(hidden_features=256, hidden_layers=2)
    g.cuda()
    f = Siren(in_features=2, out_features=3, hidden_features=256, 
              hidden_layers=4, outermost_linear=True)
    f.cuda()
    optim = torch.optim.Adam(lr=1e-4, params=chain(g.parameters(), f.parameters()))

    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 4
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xy, t = model_input[start:end, :-1], model_input[start:end, [-1]]
        # breakpoint()
        h = g(t)
        o = f(apply_homography(xy, h))
        loss = ((o - ground_truth[start:end]) ** 2).mean()

        if verbose and not step % steps_til_summary:
            print("Step [%04d/%04d]: loss=%0.4f" % (step, total_steps, loss))

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    return f, g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help="your scene name")
    parser.add_argument('--separate_num', '-sn', type=str, default='1', required=False)
    args = parser.parse_args()

    name = args.name
    sn = args.separate_num
    f, g = train_homography('data/%s/%s_all'%(name, name), 3000)

    os.makedirs(f'./output/{name}/separate_{sn}/pth_file', exist_ok=True)
    os.makedirs(f'./output/{name}/separate_{sn}/original_canonical', exist_ok=True)
    torch.save(f.state_dict(), f'./output/{name}/separate_{sn}/pth_file/homography_f.pth')
    torch.save(g.state_dict(), f'./output/{name}/separate_{sn}/pth_file/homography_g.pth')
    
    with torch.no_grad():
        xy = get_mgrid([512, 1024], [-1.5, -2.0], [1.5, 2.0]).cuda()
        output = f(xy)
        output = output.view(512, 1024, 3).cpu().detach().numpy()
        output = np.clip(output, -1, 1) * 0.5 + 0.5
        output = Image.fromarray(np.uint8(output * 255))
        output.save(f'./output/{name}/separate_{sn}/original_canonical/homography_canonical.png')