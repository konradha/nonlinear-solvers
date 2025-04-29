import os
import sys
import glob
from pathlib import Path

def find_h5_files(base_dir, output_file):
    base_path = Path(base_dir)
    h5_files = []
    
    include_prefixes = [
        "kge_kink_field_", 
        # "nlse_multi_soliton_state_", 
        # "nlse_skyrmion_tube_"
    ]
    
    for prefix in include_prefixes:
        pattern = base_path / f"{prefix}*/hdf5/*.h5"
        found = glob.glob(str(pattern))
        h5_files.extend(found)
        
    h5_files = sorted(list(set(h5_files)))
    
    if not h5_files:
        print(f"Warning: No HDF5 files found in specified subdirectories of {base_dir}", file=sys.stderr)
        
    with open(output_file, 'w') as f:
        for file_path in h5_files:
            f.write(f"{file_path}\n")
            
    print(f"Found {len(h5_files)} HDF5 files. List saved to {output_file}")
    return len(h5_files)

if __name__ == "__main__":
    scratch_dir = os.environ.get("SCRATCH")
    if not scratch_dir:
        print("Error: SCRATCH environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(scratch_dir):
         print(f"Error: SCRATCH directory '{scratch_dir}' not found or not a directory.", file=sys.stderr)
         sys.exit(1)

    list_file = "h5_file_list.txt" 
    num_files = find_h5_files(scratch_dir, list_file)
    
    if num_files == 0:
        sys.exit(1) 
        
    sys.exit(0) 
