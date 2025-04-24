import rasterio
import numpy as np
import glob
import os
from pathlib import Path

def check_tif_range(directory, num_files=10):
    """
    Check the value range of TIF files in the specified directory.
    
    Args:
        directory (str): Path to directory containing TIF files
        num_files (int): Number of files to check (default: 10)
    
    Returns:
        tuple: (min_value, max_value, mean_value, std_value)
    """
    # Find all TIF files
    tif_files = sorted(glob.glob(os.path.join(directory, '*.TIF')))
    
    if not tif_files:
        raise FileNotFoundError(f"No .TIF files found in directory: {directory}")
    
    print(f"Found {len(tif_files)} TIF files. Checking {min(num_files, len(tif_files))} files...")
    
    # Initialize lists to store statistics
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    
    # Process each file
    for i, file_path in enumerate(tif_files[:num_files]):
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                data = src.read().astype(np.float32)
                
                # Calculate statistics
                min_val = np.min(data)
                max_val = np.max(data)
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                print(f"\nFile {i+1}: {os.path.basename(file_path)}")
                print(f"  Min: {min_val:.4f}")
                print(f"  Max: {max_val:.4f}")
                print(f"  Mean: {mean_val:.4f}")
                print(f"  Std: {std_val:.4f}")
                
                all_mins.append(min_val)
                all_maxs.append(max_val)
                all_means.append(mean_val)
                all_stds.append(std_val)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_mins:
        raise ValueError("No files were successfully processed")
    
    # Calculate overall statistics
    overall_min = np.min(all_mins)
    overall_max = np.max(all_maxs)
    overall_mean = np.mean(all_means)
    overall_std = np.mean(all_stds)
    
    print("\n=== Overall Statistics ===")
    print(f"Overall Min: {overall_min:.4f}")
    print(f"Overall Max: {overall_max:.4f}")
    print(f"Overall Mean: {overall_mean:.4f}")
    print(f"Overall Std: {overall_std:.4f}")
    print("=========================")
    
    return overall_min, overall_max, overall_mean, overall_std

if __name__ == "__main__":
    # Update these paths to match your dataset structure
    lr_dir = "/home/bhargavp22co/SRGAN/SRGAN/Dataset/train_lr/"
    hr_dir = "/home/bhargavp22co/SRGAN/SRGAN/Dataset/train_hr/"
    
    print("\nChecking LR images...")
    lr_min, lr_max, lr_mean, lr_std = check_tif_range(lr_dir)
    
    print("\nChecking HR images...")
    hr_min, hr_max, hr_mean, hr_std = check_tif_range(hr_dir)
    
    print("\n=== Recommended Scaling Values ===")
    print("For LR images:")
    print(f"min_value = {lr_min:.4f}")
    print(f"max_value = {lr_max:.4f}")
    print("\nFor HR images:")
    print(f"min_value = {hr_min:.4f}")
    print(f"max_value = {hr_max:.4f}") 