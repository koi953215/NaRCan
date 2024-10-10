import os
import argparse
import numpy as np
from itertools import chain
from PIL import Image, ImageFilter

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from models.model import Homography, Siren
from util import get_mgrid, apply_homography, jacobian, VideoFitting, TestVideoFitting

from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDIMScheduler
)
from transformers import CLIPTextModel



# device1 for our model
device1 = 'cuda:0'
# device2 for diffusion model
device2 = 'cuda:1'

def get_canonical(canonical, step, output_dir):
    canonical = canonical.view(512, 512, 3)
    canonical = canonical.permute(2, 0, 1)
    canonical = torch.clip(canonical, -1, 1) * 0.5 + 0.5
    if step % 1000 == 0:
        canonical_save = canonical.detach().clone().permute(1, 2, 0).cpu().numpy()
        canonical_save = Image.fromarray(np.uint8(canonical_save * 255))
        canonical_save.save(os.path.join(output_dir, 'training_canonicals', 'canonical_%d.png'%step))

    return canonical

def train_residual_flow(path, total_steps, lambda_flow=0.02, verbose=True, steps_til_summary=100):
    global pipe, output_dir, generator

    transform = Compose([
        Resize(512),
        ToTensor(),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])
    v = VideoFitting(path, transform)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)

    g = Siren(in_features=3, out_features=2, hidden_features=256,
              hidden_layers=5, outermost_linear=True)
    g.cuda()
    f = Siren(in_features=2, out_features=3, hidden_features=256, 
            hidden_layers=5, outermost_linear=True)
    f.cuda()

    optim = torch.optim.Adam(lr=1e-4, params=chain(g.parameters(), f.parameters()))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[4000, 8000, 10000, 12500], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 8000, 10000], gamma=0.1)

    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 32
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))
        
        xy, t = model_input[start:end, :-1], model_input[start:end, [-1]]
        xyt = model_input[start:end].requires_grad_()
        # breakpoint()

        h_old = apply_homography(xy, g_old(t))
        h = g(xyt)
        xy_ = h_old + h
        o = f(xy_)
        loss_recon = (o - ground_truth[start:end]).abs().mean()
        loss_flow = jacobian(h, xyt).abs().mean()
        loss = loss_recon + lambda_flow * loss_flow

        # set dm render frequency and strength
        if step <= 3000:
            dm_freq = 10
            dm_strength = 0.4
        elif step <= 5000:
            dm_freq = 100
            dm_strength = 0.3
        else:
            dm_freq = 2000
            dm_strength = 0.2

        loss_dm = 0
        # start to join diffusion prior
        if step >= 1000:
            xy_c = get_mgrid([512, 512], [cx_min, cy_min], [cx_max, cy_max]).to(device1)
            o_c = f(xy_c)
            # o_c shape: (C, H, W)
            o_c = get_canonical(o_c, step, output_dir)
            # dilated_mask shape: (C, H, W)
            dilated_mask = torch.ones((1, 512, 512), dtype=torch.float32).to(device1)
            # use pre-trained diffusion model
            # image shape: (B, C, H, W) or (C, H, W)
            # image value: 0~1
            if step % dm_freq == 0:
                dm_result = pipe(
                    ["a photo of sks"] * 1, image=o_c.detach().clone().to(device2), mask_image=dilated_mask.detach().clone().to(device2), 
                    num_inference_steps=60, guidance_scale=1, generator=generator, strength=dm_strength
                ).images
                # conver tensor to image
                o_image = o_c.detach().clone().permute(1, 2, 0).cpu().numpy()
                o_image = Image.fromarray(np.uint8(o_image * 255))

                mask_image = dilated_mask.detach().clone().cpu().numpy().squeeze()
                mask_image = Image.fromarray(np.uint8(mask_image * 255))

                erode_kernel = ImageFilter.MaxFilter(3)
                mask_image = mask_image.filter(erode_kernel)
                
                blur_kernel = ImageFilter.BoxBlur(1)
                mask_image = mask_image.filter(blur_kernel)

                for idx, result in enumerate(dm_result):
                    result = Image.composite(result, o_image, mask_image)
                    if step % 1000 == 0:
                        result.save(os.path.join(output_dir, 'dm_results', 'result_%d.png'%step))
                
                result = torch.tensor(np.array(result), dtype=torch.float32).to(device1) / 255.0
                result = result.permute(2, 0, 1)
            
            # compute DM MSE loss
            myweight = torch.zeros(512, 512, dtype=torch.float32).to(device1)
            myweight[:, :] = 1.0
            loss_dm += (myweight * (o_c - result)).abs().mean()
            # loss_dm += (o_c - result).abs().mean()

            loss += loss_dm

        if verbose and not step % steps_til_summary:
            print("Step [%04d/%04d]: recon=%0.8f, flow=%0.4f, dm=%.05f" % (step, total_steps, loss_recon, loss_flow, loss_dm))

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
    
    return f, g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help="your scene name")
    parser.add_argument('--diffusion_path', '-dp', type=str, required=True, help="the path of your diffusion model.")
    parser.add_argument('--separate_num', '-sn', type=str, default='1', required=False)
    args = parser.parse_args()

    name = args.name
    sn = args.separate_num
    model_path = args.diffusion_path

    # set global variables
    global g_old, pipe, output_dir, generator
    global cx_max, cx_min, cy_max, cy_min

    # Load homography checkpoints
    checkpoint_g_old = torch.load('output/%s/separate_%s/pth_file/homography_g.pth'%(name, sn))
    g_old = Homography(hidden_features=256, hidden_layers=2).cuda()
    g_old.load_state_dict(checkpoint_g_old)
    g_old.eval()
    print("---Loading successfully---")

    # calcaulate real canonical region
    transform = Compose([
        Resize(512),
        ToTensor(),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])
    v = TestVideoFitting('data/%s/%s_all'%(name, name), transform)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)

    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].to(device1), ground_truth[0].to(device1)

    data_len = len(os.listdir('data/%s/%s_all'%(name, name)))
    real_area = []

    with torch.no_grad():
        batch_size = (v.H * v.W)
        for step in range(data_len):
            start = (step * batch_size) % len(model_input)
            end = min(start + batch_size, len(model_input))

            xy, t = model_input[start:end, :-1], model_input[start:end, [-1]]
            h_old = apply_homography(xy, g_old(t))
            xy_ = h_old
            real_area.append(xy_)
        
        real_area = torch.stack(real_area)
        real_area = real_area.reshape(-1, 2)
        cx_max, cx_min = torch.max(real_area[:, 0]), torch.min(real_area[:, 0])
        cy_max, cy_min = torch.max(real_area[:, 1]), torch.min(real_area[:, 1])

        save_x = round(max(abs(cx_max.item()), abs(cx_min.item())), 3)
        save_y = round(max(abs(cy_max.item()), abs(cy_min.item())), 3)

        with open('output/%s/separate_%s/canonical_region.txt'%(name, sn), 'w') as ff:
            ff.write('canonical_region\n')
            ff.write(str(save_x) + "\n")
            ff.write(str(save_y) + "\n")

    generator = None 
    seed = None
    output_dir = 'output/%s/separate_%s/training_log'%(name, sn)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'training_canonicals'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dm_results'), exist_ok=True)

    # create & load model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
        revision=None
    )

    pipe.unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", revision=None,
    )
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", revision=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device2)

    if seed is not None:
        generator = torch.Generator(device=device2).manual_seed(seed)
    
    for k in range(1, int(sn) + 1):
        now_path = f"data/{name}/separate_{sn}/{name}_{k}"
        f, g = train_residual_flow(now_path, 12500, lambda_flow=0.03)
        
        torch.save(f.state_dict(), f'./output/{name}/separate_{sn}/pth_file/mlp_f{k}.pth')
        torch.save(g.state_dict(), f'./output/{name}/separate_{sn}/pth_file/mlp_g{k}.pth')
        
        with torch.no_grad():
            xy = get_mgrid([512, 1024], [-save_x, -save_y], [save_x, save_y]).cuda()
            output = f(xy)
            output = output.view(512, 1024, 3).cpu().detach().numpy()
            output = np.clip(output, -1, 1) * 0.5 + 0.5
            output = Image.fromarray(np.uint8(output * 255))
            output.save(f'./output/{name}/separate_{sn}/original_canonical/canonical_{k}.png')
   

if __name__ == '__main__':
    main()