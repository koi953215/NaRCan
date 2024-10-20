import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import argparse
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import math

def make_image_grid_from_folder(image_files, separate_num, image_size=(512, 1024), save_grid=False):
    transform = transforms.Compose([
        transforms.Resize(image_size),  
        transforms.ToTensor()   
    ])
    
    images = [transform(Image.open(os.path.join(folder_path, img_file))) for img_file in image_files]
    grid = make_grid(images, nrow=math.ceil(separate_num/2), padding=0)
    
    if save_grid:
        save_image(grid, os.path.join(folder_path, "merge_canonical.png"))
    return grid
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help='name')
    parser.add_argument('--separate_num', '-sn', type=int, default=3, help='Number of separations')
    args = parser.parse_args()
    
    assert args.separate_num >= 2 and args.separate_num <=6
    folder_path = os.path.join("output", args.name, f"separate_{args.separate_num}", "original_canonical")
    image_files = [f"canonical_{sep}.png" for sep in range(1, args.separate_num+1)]
    
    grid = make_image_grid_from_folder(image_files, args.separate_num, save_grid=True)
    os.makedirs(os.path.join("output", args.name, f"separate_{args.separate_num}", "edited_canonical"), exist_ok=True)