import os
import glob
import numpy as np
import tensorflow as tf
import rasterio
from sklearn.model_selection import train_test_split
from tensorflow.python.data.experimental import AUTOTUNE
import random # for sampling

def load_and_preprocess_image(file_path_tensor):
    """Loads and preprocesses a single TIF image using rasterio."""
    
    def _load_image(file_path):
        file_path = file_path.numpy().decode('utf-8')
        with rasterio.open(file_path) as src:
            img = src.read().transpose((1, 2, 0))
            img = img.astype(np.float32)
            ACTUAL_DATA_MIN = 0.0191
            ACTUAL_DATA_MAX = 0.4274
            img = (img - ACTUAL_DATA_MIN) / (ACTUAL_DATA_MAX - ACTUAL_DATA_MIN)
            img = img * 255.0
            img = np.clip(img, 0.0, 255.0)
        return img

    img = tf.py_function(
        _load_image,
        [file_path_tensor],
        tf.float32
    )
    return img

def get_oli2msi_paths(datadir):
    """Finds paired LR and HR image paths within train_lr/ and train_hr/ subdirs."""
    lr_dir = os.path.join(datadir, 'train_lr') 
    hr_dir = os.path.join(datadir, 'train_hr')

    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"Training LR directory not found: {lr_dir}")
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"Training HR directory not found: {hr_dir}")

    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.TIF')))
    if not lr_files:
        raise FileNotFoundError(f"No .TIF files found in the training LR directory: {lr_dir}")

    hr_files = []
    valid_lr_files = [] 
    missing_hr_count = 0
    hr_file_basenames = {os.path.basename(p) for p in glob.glob(os.path.join(hr_dir, '*.TIF'))}

    for lr_path in lr_files:
        filename = os.path.basename(lr_path)
        hr_filename = filename 
        if hr_filename in hr_file_basenames:
            hr_path = os.path.join(hr_dir, hr_filename)
            if os.path.exists(hr_path):
                valid_lr_files.append(lr_path)
                hr_files.append(hr_path)
            else:
                 missing_hr_count += 1
                 print(f"Warning: HR file {hr_filename} found in listing but path not valid: {hr_path}")
        else:
            missing_hr_count += 1
            
    if missing_hr_count > 0:
        print(f"Warning: Found {len(lr_files)} LR files, but {missing_hr_count} were missing corresponding HR files in {hr_dir}. Only using pairs with found HR files.")
        
    lr_files = valid_lr_files 

    if not lr_files:
         raise ValueError(f"No matched LR/HR pairs found between {lr_dir} and {hr_dir}. Check filenames are identical and files exist.")

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

    if set_shapes:
        lr_ds = lr_ds.map(lambda x: tf.ensure_shape(x, [160, 160, 3]))
        hr_ds = hr_ds.map(lambda x: tf.ensure_shape(x, [480, 480, 3]))

    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_oli2msi_datasets(datadir, batch_size=16, val_split=0.2, random_state=42, num_pairs_to_use=3000):
    """Creates training and validation datasets for OLI2MSI, using a subset of pairs."""
    
    all_lr_paths, all_hr_paths = get_oli2msi_paths(datadir)

    if not all_lr_paths:
        return None, None

    num_available_pairs = len(all_lr_paths)
    if num_available_pairs > num_pairs_to_use:
        print(f"Sampling {num_pairs_to_use} pairs from {num_available_pairs} available pairs...")
        indices = list(range(num_available_pairs))
        if random_state is not None:
            random.seed(random_state) 
        sampled_indices = random.sample(indices, num_pairs_to_use)
        lr_paths = [all_lr_paths[i] for i in sampled_indices]
        hr_paths = [all_hr_paths[i] for i in sampled_indices]
        print(f"Using {len(lr_paths)} sampled pairs.")
    else:
        print(f"Using all {num_available_pairs} available pairs (less than or equal to requested {num_pairs_to_use}).")
        lr_paths = all_lr_paths
        hr_paths = all_hr_paths

    if val_split > 0 and len(lr_paths) > 1:
        lr_train_paths, lr_val_paths, hr_train_paths, hr_val_paths = train_test_split(
            lr_paths, hr_paths, test_size=val_split, random_state=random_state, shuffle=True
        )
    elif len(lr_paths) == 1 and val_split > 0:
         print("Only 1 image pair available after sampling. Cannot create validation split. Using the pair for training.")
         lr_train_paths, hr_train_paths = lr_paths, hr_paths
         lr_val_paths, hr_val_paths = [], []
    else: 
        lr_train_paths, hr_train_paths = lr_paths, hr_paths
        lr_val_paths, hr_val_paths = [], []

    print(f"Total pairs used for split: {len(lr_paths)}")
    print(f"Training set size: {len(lr_train_paths)}")
    print(f"Validation set size: {len(lr_val_paths)}")

    if not lr_train_paths:
        print("Error: No training data available after sampling and splitting.")
        return None, None

    train_ds = create_dataset(lr_train_paths, hr_train_paths, batch_size)

    val_ds = None
    if lr_val_paths:
        val_ds = create_dataset(lr_val_paths, hr_val_paths, batch_size=1, set_shapes=True)

    return train_ds, val_ds

