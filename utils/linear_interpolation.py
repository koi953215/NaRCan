import os
import numpy as np
import imageio.v2 as imageio
import argparse

def blend_images(img_files, img, denominator, i):
    """Blend two images based on the weight from their index."""
    img_files[-(denominator-i-1)] = img_files[-(denominator-i-1)] * (1 - (i+1)/denominator) + img * ((i+1)/denominator)

def read_image(input_path):
    img = imageio.imread(input_path)
    return img.astype(np.float64)

def read_images_in_folder(img_files, input_path):
    """Read and process all images in the input path."""
    for img_path in sorted(os.listdir(input_path)):
        img_files.append(read_image(f"{input_path}/{img_path}"))
    return img_files

def linear_interpolate(name, separation_path, output_path, n_separation, fps=15, save_video=False):
    overlap_path = os.path.join(separation_path, f"{name}_1")
    img_files = []
    img_files = read_images_in_folder(img_files, overlap_path)

    if n_separation != 1:
        next_overlap_path = os.path.join(separation_path, f"{name}_2")
        prev_last = int(sorted(os.listdir(overlap_path))[-1].split('.')[0])
        now_first = int(sorted(os.listdir(next_overlap_path))[0].split('.')[0])
        denominator = prev_last - now_first + 2
        

        for s_ in range(2, n_separation):
            print(f"=== Overlap {s_-1}-{s_} ===")
            overlap_path = next_overlap_path
            next_overlap_path = os.path.join(separation_path, f"{name}_{s_+1}")
            
            for i, img_path in enumerate(sorted(os.listdir(overlap_path))):
                now_idx = int(img_path.split('.')[0])
                img = read_image(f"{overlap_path}/{img_path}")
                
                if now_idx <= prev_last:
                    # print(now_idx, img_path, ((i+1)/denominator), (1 - (i+1)/denominator))
                    blend_images(img_files, img, denominator, i)
                else:
                    img_files.append(img)

            prev_last = int(sorted(os.listdir(overlap_path))[-1].split('.')[0])
            now_first = int(sorted(os.listdir(next_overlap_path))[0].split('.')[0])
            denominator = prev_last - now_first + 2

        # Process last overlap
        print(f"=== Overlap {n_separation-1}-{n_separation} ===")
        overlap_path = next_overlap_path
        for i, img_path in enumerate(sorted(os.listdir(overlap_path))):
            now_idx = int(img_path.split('.')[0])
            img = imageio.imread(f"{overlap_path}/{img_path}")
            if now_idx <= prev_last:
                blend_images(img_files, img, denominator, i)
            else:
                img_files.append(img)

    # Save final images
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, f"{name}_result"), exist_ok=True)
    print(f"Save {len(img_files)} images in {output_path}/{name}_result")
    img_files = np.asarray(img_files).astype(np.uint8)
    for i, img in enumerate(img_files):
        imageio.imwrite(os.path.join(output_path, f"{name}_result", f'{i:05d}.png'), img)
        
    if save_video:
        print(f"Save video images in {output_path}/{name}_result.mp4")
        writer = imageio.get_writer(f"{output_path}/{name}_result.mp4", fps=fps)
        for img in img_files:
            writer.append_data(img)
        writer.close()