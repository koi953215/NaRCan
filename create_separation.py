import os
import shutil
import argparse

def copy_files_with_overlap(args):
    """
    Copy files from the source directory to the target directory in separate_num segments 
    with specified overlap between segments.
    """

    root_path = os.path.join("data", f"{args.name}")
    from_path = os.path.join(root_path, f"{args.name}_all")
    to_path = os.path.join(root_path, f"separate_{args.separate_num}")

    # List and sort the files in the source directory
    file_list = sorted(os.listdir(from_path))
    data_len = len(file_list)
    
    # Calculate number of files per segment
    n_segment = data_len // args.separate_num

    # Copy files for the middle segments
    for segment in range(1, args.separate_num + 1):
        # Create directory
        to_path_complete = os.path.join(to_path, f"{args.name}_{segment}")
        os.makedirs(to_path_complete, exist_ok=True)

        if args.separate_num == 1:
            start_idx = 0
            end_idx = data_len
        elif segment == 1:
            start_idx = 0
            end_idx = n_segment + (args.overlap_num // 2 + args.overlap_num % 2)
        elif segment == args.separate_num:
            start_idx = n_segment * (segment - 1) - (args.overlap_num // 2)
            end_idx = data_len
        else:
            start_idx = n_segment * (segment - 1) - (args.overlap_num // 2)
            end_idx = n_segment * segment + (args.overlap_num // 2 + args.overlap_num % 2)

        print(start_idx, end_idx)
        for i in range(start_idx, end_idx):
            shutil.copy(f'{from_path}/{file_list[i]}', f'{to_path_complete}/{file_list[i]}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create separation.")
    parser.add_argument('--name', '-n', type=str, required=True, help='name')
    parser.add_argument('--separate_num', '-sn', type=int, default=3, help='Number of separations to split the files into.')
    parser.add_argument('--overlap_num', '-on', type=int, default=10, help='Number of overlapping files between consecutive segments.')

    args = parser.parse_args()

    copy_files_with_overlap(args)