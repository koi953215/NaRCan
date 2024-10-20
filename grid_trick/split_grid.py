import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import argparse
import numpy as np
import math


def split_grid_into_images(folder_path, separate_num, image_size=(512, 1024), padding=0):
    grid_image = Image.open(os.path.join(folder_path, "merge_canonical.png"))
    grid_image = np.asarray(grid_image)
    
    nrow = math.ceil(separate_num/2)
    ncol = separate_num-nrow
    if ncol == 1:
        ncol = 2
    cnt = 0
        
    for i in range(nrow):
        for j in range(ncol):
            if cnt >= separate_num:
                break
            
            cnt += 1
            img = grid_image[image_size[0]*(i): image_size[0]*(i+1), image_size[1]*(j):image_size[1]*(j+1), :]
            output_path = os.path.join(folder_path, f"canonical_{cnt}.png")
            pil_img = Image.fromarray(img)
            pil_img = pil_img
            pil_img.save(output_path)
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help='name')
    parser.add_argument('--separate_num', '-sn', type=int, default=3, help='Number of separations')
    args = parser.parse_args()
    
    assert args.separate_num >= 2 and args.separate_num <=6
    folder_path = os.path.join("output", args.name, f"separate_{args.separate_num}", "edited_canonical")
    split_grid_into_images(folder_path, args.separate_num)