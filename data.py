import os
import glob
import numpy as np
import tensorflow as tf
import rasterio
from sklearn.model_selection import train_test_split
from tensorflow.python.data.experimental import AUTOTUNE
import random # Import random for sampling

def load_and_preprocess_image(file_path_tensor):
    """Loads and preprocesses a single TIF image using rasterio."""
    
    def _load_image(file_path):
        file_path = file_path.numpy().decode('utf-8')
        with rasterio.open(file_path) as src:
            # Read image bands, transpose to (height, width, channels)
            img = src.read().transpose((1, 2, 0))
            
            # Ensure float32 type
            img = img.astype(np.float32)
            
            # --- Scaling to [0, 255] ---
            # Define actual data range (adjust if needed based on dataset inspection)
            ACTUAL_DATA_MIN = 0.0191
            ACTUAL_DATA_MAX = 0.4274

            # Scale to [0, 1] using min-max normalization
            img = (img - ACTUAL_DATA_MIN) / (ACTUAL_DATA_MAX - ACTUAL_DATA_MIN)

            # Scale to [0, 255]
            img = img * 255.0

            # Clip to ensure the range is strictly [0, 255]
            img = np.clip(img, 0.0, 255.0)
            # --- End Scaling ---
            
        return img

    # Wrap the rasterio loading logic in tf.py_function
    img = tf.py_function(
        _load_image,
        [file_path_tensor],
        tf.float32
    )
    
    # Let TF infer shape for now, but explicitly setting is safer if fixed.
    # Example: img.set_shape([None, None, 3]) 
    
    return img

def get_oli2msi_paths(datadir):
    """Finds paired LR and HR image paths within train_lr/ and train_hr/ subdirs."""
    # Point to the correct subdirectories
    lr_dir = os.path.join(datadir, 'train_lr') 
    hr_dir = os.path.join(datadir, 'train_hr')

    # Check if directories exist
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"Training LR directory not found: {lr_dir}")
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"Training HR directory not found: {hr_dir}")

    # Find all TIF files in the training LR directory
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.TIF')))
    if not lr_files:
        # Adjusted error message for clarity
        raise FileNotFoundError(f"No .TIF files found in the training LR directory: {lr_dir}")

    hr_files = []
    valid_lr_files = [] # Keep track of LR files that have a corresponding HR file
    missing_hr_count = 0
    
    # Create a set of HR filenames for efficient lookup
    # Use glob to find all .TIF files in hr_dir first
    hr_file_basenames = {os.path.basename(p) for p in glob.glob(os.path.join(hr_dir, '*.TIF'))}

    for lr_path in lr_files:
        filename = os.path.basename(lr_path)
        # Assume HR filename is identical to LR filename
        hr_filename = filename 
        
        # Check if the identical filename exists in the set of HR filenames
        if hr_filename in hr_file_basenames:
            hr_path = os.path.join(hr_dir, hr_filename)
            # Final check to ensure the path actually exists (though set check should suffice)
            if os.path.exists(hr_path):
                valid_lr_files.append(lr_path)
                hr_files.append(hr_path)
            else:
                 # This case should be rare if hr_file_basenames was built correctly
                 missing_hr_count += 1
                 print(f"Warning: HR file {hr_filename} found in listing but path not valid: {hr_path}")
        else:
            missing_hr_count += 1
            # Optional: print(f"Warning: Missing corresponding HR file for {lr_path} (expected {hr_filename} in {hr_dir})") 
            
    if missing_hr_count > 0:
        print(f"Warning: Found {len(lr_files)} LR files, but {missing_hr_count} were missing corresponding HR files in {hr_dir}. Only using pairs with found HR files.")
        
    lr_files = valid_lr_files # Update lr_files to only include those with matches

    if not lr_files:
         # Adjusted error message
         raise ValueError(f"No matched LR/HR pairs found between {lr_dir} and {hr_dir}. Check filenames are identical and files exist.")

    # Ensure the number of found pairs match before returning
    if len(lr_files) != len(hr_files):
        raise RuntimeError(f"Internal Error: Mismatch between valid LR ({len(lr_files)}) and HR ({len(hr_files)}) lists after filtering.")

    print(f"Found {len(lr_files)} corresponding LR/HR image pairs in train_lr/ and train_hr/.")
    return lr_files, hr_files


def create_dataset(lr_paths, hr_paths, batch_size, set_shapes=True):
    """Creates a tf.data.Dataset from LR and HR image paths."""
    lr_ds = tf.data.Dataset.from_tensor_slices(lr_paths)
    hr_ds = tf.data.Dataset.from_tensor_slices(hr_paths)

    lr_ds = lr_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    hr_ds = hr_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    # Set shapes explicitly if they are fixed (recommended)
    if set_shapes:
        lr_ds = lr_ds.map(lambda x: tf.ensure_shape(x, [160, 160, 3]))
        hr_ds = hr_ds.map(lambda x: tf.ensure_shape(x, [480, 480, 3]))

    ds = tf.data.Dataset.zip((lr_ds, hr_ds))

    # Note: Removed random transformations (crop, flip, rotate) 
    # Re-add if data augmentation is desired for satellite images.

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_oli2msi_datasets(datadir, batch_size=16, val_split=0.2, random_state=42, num_pairs_to_use=3000):
    """Creates training and validation datasets for OLI2MSI, using a subset of pairs."""
    
    all_lr_paths, all_hr_paths = get_oli2msi_paths(datadir)

    if not all_lr_paths:
        # Error raised in get_oli2msi_paths if no pairs found
        return None, None

    # --- Limit dataset size --- 
    num_available_pairs = len(all_lr_paths)
    if num_available_pairs > num_pairs_to_use:
        print(f"Sampling {num_pairs_to_use} pairs from {num_available_pairs} available pairs...")
        # Create indices and sample using random.sample for efficiency
        indices = list(range(num_available_pairs))
        # Use random_state for reproducibility if provided
        if random_state is not None:
            random.seed(random_state) 
        sampled_indices = random.sample(indices, num_pairs_to_use)
        # Retrieve the sampled paths
        lr_paths = [all_lr_paths[i] for i in sampled_indices]
        hr_paths = [all_hr_paths[i] for i in sampled_indices]
        print(f"Using {len(lr_paths)} sampled pairs.")
    else:
        print(f"Using all {num_available_pairs} available pairs (less than or equal to requested {num_pairs_to_use}).")
        lr_paths = all_lr_paths
        hr_paths = all_hr_paths
    # --- End dataset size limit ---

    # Now proceed with the train/validation split using the (potentially limited) lr_paths and hr_paths
    if val_split > 0 and len(lr_paths) > 1:
        lr_train_paths, lr_val_paths, hr_train_paths, hr_val_paths = train_test_split(
            lr_paths, hr_paths, test_size=val_split, random_state=random_state, shuffle=True
        )
    elif len(lr_paths) == 1 and val_split > 0:
         print("Warning: Only 1 image pair available after sampling. Cannot create validation split. Using the pair for training.")
         lr_train_paths, hr_train_paths = lr_paths, hr_paths
         lr_val_paths, hr_val_paths = [], []
    else: # val_split == 0 or only 1 sample
        lr_train_paths, hr_train_paths = lr_paths, hr_paths
        lr_val_paths, hr_val_paths = [], []

    print(f"Total pairs used for split: {len(lr_paths)}")
    print(f"Training set size: {len(lr_train_paths)}")
    print(f"Validation set size: {len(lr_val_paths)}")

    # Ensure train set is not empty before creating dataset
    if not lr_train_paths:
        print("Error: No training data available after sampling and splitting.")
        return None, None

    train_ds = create_dataset(lr_train_paths, hr_train_paths, batch_size)

    val_ds = None
    if lr_val_paths: # Only create validation dataset if paths exist
        # Using batch_size=1 for validation dataset for easier image saving in callback
        val_ds = create_dataset(lr_val_paths, hr_val_paths, batch_size=1, set_shapes=True)

    return train_ds, val_ds

# --- Removed DIV2K specific class and download functions ---
# --- Removed original transformations (random_crop, random_flip, random_rotate) ---
