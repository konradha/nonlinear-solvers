import os
import sys
import glob
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)

def find_nan_containing_files(base_dir, output_file, pattern="**/*.h5"):
    base_path = Path(base_dir)
    found_files = list(base_path.glob(pattern))
    found_files = sorted(list(set(found_files)))
    
    nan_files = []
    total_files = len(found_files)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print("Using device=", device)
    for i, h5_file in enumerate(found_files): 
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'u' in f:
                    u_dataset = np.array(f['u'])
                    u_torch = torch.tensor(u_dataset).to(device)
                    if torch.isnan(u_torch).any().item():
                        print(str(h5_file), "seems useless")
                        nan_files.append(str(h5_file))
                        del u_dataset
                        if device == torch.device("cuda"):     
                            del u_torch
                            torch.cuda.empty_cache()
                        continue  
                    # let's not run into issues by exploding GPU mem
                    del u_dataset
                    if device == torch.device("cuda"):
                        del u_torch
                        torch.cuda.empty_cache()
                    

        except Exception as e:
            # maybe file writing has been interrupted if time limit on SLURM reached
            print(f"Error processing {h5_file}: {e}")
            nan_files.append(str(h5_file) + f" # Error: {e}")
    
    print(f"Found {len(nan_files)} files containing NaN values.")
    print(f"{((found_files / len(nan_files) * 100.)):.2f} %")
   
    with open(output_file, 'w') as f:
        for file_path in nan_files:
            f.write(f"{file_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Find HDF5 files with NaN values in 'u' dataset.")
    parser.add_argument("base_dir", type=str, help="Base directory containing HDF5 files.")
    parser.add_argument("output_file", type=str, help="Path to output text file listing invalid files.")
    parser.add_argument("--pattern", type=str, default="**/*.h5", help="Glob pattern for finding HDF5 files.")
    
    args = parser.parse_args()
    find_nan_containing_files(args.base_dir, args.output_file, args.pattern)

if __name__ == "__main__":
    print("Let's remove unusable data")
    main()
