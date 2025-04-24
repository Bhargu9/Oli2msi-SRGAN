import os
import argparse
import glob
import time
import numpy as np
import tensorflow as tf
import rasterio
from PIL import Image # For saving PNG outputs
import tifffile # Optional: for saving TIF outputs
from skimage.metrics import structural_similarity as ssim # For SSIM calculation

# Assuming model/srgan.py and model/common.py are in the same directory
# or accessible via Python path
from model.srgan import generator
from model.common import psnr

# Constants for data scaling
ACTUAL_DATA_MIN = 0.0191
ACTUAL_DATA_MAX = 0.4274

def load_and_preprocess_lr(lr_path):
    """
    Loads an LR TIF image, scales it from its raw range to [0, 255] float32,
    and adds a batch dimension.
    """
    try:
        with rasterio.open(lr_path) as src:
            # Read image bands, transpose to (height, width, channels)
            img = src.read().transpose((1, 2, 0))
            img = img.astype(np.float32)

            # --- Scaling raw values (~[0.0191, 0.4274]) to [0, 255] float32 ---
            # 1. Scale to [0, 1] using min-max normalization
            img_norm_01 = (img - ACTUAL_DATA_MIN) / (ACTUAL_DATA_MAX - ACTUAL_DATA_MIN)
            # 2. Scale to [0, 255]
            img_scaled_255 = img_norm_01 * 255.0
            # 3. Clip to ensure the range is strictly [0.0, 255.0]
            img_clipped = np.clip(img_scaled_255, 0.0, 255.0)
            # --- End Scaling ---

            # Add batch dimension
            img_batch = tf.expand_dims(img_clipped, axis=0)
            return img_batch
    except Exception as e:
        print(f"Error loading/preprocessing LR image {lr_path}: {e}")
        return None

def load_hr_for_eval(hr_path):
    """
    Loads an HR TIF image, scales it to [0, 255] float32,
    and converts it to uint8 NumPy array for evaluation.
    """
    try:
        with rasterio.open(hr_path) as src:
            # Read image bands, transpose to (height, width, channels)
            img = src.read().transpose((1, 2, 0))
            img = img.astype(np.float32)

            # --- Scaling raw values (~[0.0191, 0.4274]) to [0, 255] float32 ---
            img_norm_01 = (img - ACTUAL_DATA_MIN) / (ACTUAL_DATA_MAX - ACTUAL_DATA_MIN)
            img_scaled_255 = img_norm_01 * 255.0
            img_clipped = np.clip(img_scaled_255, 0.0, 255.0)
            # --- End Scaling ---

            # Convert to uint8 for evaluation
            img_uint8 = img_clipped.astype(np.uint8)
            return img_uint8
    except Exception as e:
        print(f"Error loading HR image {hr_path}: {e}")
        return None

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    print("Loading generator model...")
    try:
        model = generator() # Instantiate the generator architecture
        model.load_weights(args.weights) # Load the trained weights
        print(f"Loaded weights from {args.weights}")
    except Exception as e:
        print(f"Error loading model or weights: {e}")
        return

    # --- Find Test Image Pairs ---
    lr_files = sorted(glob.glob(os.path.join(args.lr_dir, '*.TIF')))
    if not lr_files:
        print(f"Error: No .TIF files found in LR directory: {args.lr_dir}")
        return

    test_pairs = []
    for lr_path in lr_files:
        base_filename = os.path.basename(lr_path)
        hr_path = os.path.join(args.hr_dir, base_filename)
        if os.path.exists(hr_path):
            test_pairs.append((lr_path, hr_path))
        else:
            print(f"Warning: Corresponding HR file not found for {lr_path}, skipping.")

    if not test_pairs:
        print("Error: No valid LR/HR pairs found.")
        return

    print(f"Found {len(test_pairs)} test image pairs.")

    # --- Evaluation Loop ---
    psnr_values = []
    ssim_values = []

    for i, (lr_path, hr_path) in enumerate(test_pairs):
        print(f"\nProcessing pair {i+1}/{len(test_pairs)}: {os.path.basename(lr_path)}")

        # 1. Load and preprocess LR image
        lr_input_tensor = load_and_preprocess_lr(lr_path)
        if lr_input_tensor is None:
            continue

        # 2. Generate SR image
        start_time = time.time()
        sr_output_tensor = model.predict(lr_input_tensor) # Output is [0, 255] float32
        end_time = time.time()
        print(f"  Inference time: {end_time - start_time:.4f} seconds")

        # 3. Post-process SR image for evaluation/saving (convert to uint8 NumPy)
        # Remove batch dimension and convert to NumPy
        sr_output_numpy = sr_output_tensor[0]
        # Clip and cast to uint8
        sr_output_clipped = np.clip(sr_output_numpy, 0, 255)
        sr_output_uint8 = sr_output_clipped.astype(np.uint8)

        # 4. Load HR image for evaluation
        hr_uint8 = load_hr_for_eval(hr_path)
        if hr_uint8 is None:
            continue

        # 5. Calculate Metrics
        try:
            # PSNR (using common.psnr, expects uint8)
            psnr_tf = psnr(hr_uint8, sr_output_uint8) # Ensure common.psnr handles NumPy uint8
            psnr_val = float(psnr_tf.numpy()) # Convert TF tensor result
            psnr_values.append(psnr_val)
            print(f"  PSNR: {psnr_val:.4f} dB")

            # SSIM (using skimage, expects uint8 NumPy)
            ssim_val = ssim(hr_uint8, sr_output_uint8, channel_axis=-1, data_range=255)
            ssim_values.append(ssim_val)
            print(f"  SSIM: {ssim_val:.4f}")

        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            continue

        # 6. Save Generated SR Image
        try:
            sr_filename_base = f"sr_{os.path.basename(lr_path)}"
            # Save as PNG (more common for viewing)
            sr_save_path_png = os.path.join(args.output_dir, sr_filename_base.replace('.TIF', '.png'))
            Image.fromarray(sr_output_uint8).save(sr_save_path_png)
            print(f"  Saved SR image (uint8 PNG) to: {sr_save_path_png}")

            # Optional: Save as float32 TIF for numerical analysis
            # sr_save_path_tif = os.path.join(args.output_dir, sr_filename_base)
            # tifffile.imwrite(sr_save_path_tif, sr_output_numpy.astype(np.float32), imagej=False, metadata={'axes': 'YXC'})
            # print(f"  Saved SR image (float32 TIF) to: {sr_save_path_tif}")

        except Exception as e:
            print(f"  Error saving SR image: {e}")

    # --- Calculate and Print Average Metrics ---
    if psnr_values and ssim_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print("\n--- Evaluation Summary ---")
        print(f"Average PSNR over {len(psnr_values)} images: {avg_psnr:.4f} dB")
        print(f"Average SSIM over {len(ssim_values)} images: {avg_ssim:.4f}")
    else:
        print("\nNo metrics were calculated.")

    print("\nTesting script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained SRGAN generator model.")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the generator .h5 weights file.')
    parser.add_argument('--lr_dir', type=str, required=True,
                        help='Directory containing low-resolution .TIF test images.')
    parser.add_argument('--hr_dir', type=str, required=True,
                        help='Directory containing corresponding high-resolution .TIF test images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated SR images and results.')

    args = parser.parse_args()
    main(args)