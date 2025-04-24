import os
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio # Needed indirectly by data.py
import tifffile # Import tifffile for saving TIF images
from PIL import Image # For saving PNG images

# Import model components and data loader
from model import srgan
from model.srgan import generator as create_generator
from model.srgan import discriminator as create_discriminator
from data import get_oli2msi_datasets
from model.common import normalize_m11, psnr # Added psnr import

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau # Added EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Mean

# --- Constants and Configuration ---
# VGG feature layer to use for perceptual loss
# VGG54 (original) uses layer 20 (block5_conv4)
# Try a middle layer like block3_conv4 (layer index 10) which might capture relevant textures better for satellite images
CONTENT_LOSS_LAYER = 10 # Changed from 20 to 10 (block3_conv4)

# --- Define Loss Weights (Tunable) ---
# These weights balance the contribution of different losses to the generator's objective.
# L1 and Content loss are on the [0, 255] scale, potentially large.
# Adversarial LSGAN loss is likely smaller scale.
L1_WEIGHT = 1e-2
CONTENT_WEIGHT = 6e-3 # Chosen relative to adversarial weight (1e-3)
ADVERSARIAL_WEIGHT = 1e-3

# --- Helper Functions ---

# def denormalize_image(img):
#     """Denormalizes image from [-1, 1] to [0, 255] uint8."""
#     img = (img + 1.0) * 127.5
#     return tf.clip_by_value(img, 0.0, 255.0)

# --- Custom Callback for Image Saving ---

class SaveIntermediateImages(Callback):
    def __init__(self, generator_model, validation_dataset, output_dir, save_freq=5): # Defaulting save_freq to 5 as mentioned
        super().__init__()
        self.generator = generator_model
        self.val_ds = validation_dataset # Expects batch_size=1
        self.output_dir = output_dir
        self.save_freq = save_freq
        # Define specific subdirectories for TIF and PNG outputs
        self.tif_save_dir = os.path.join(output_dir, 'intermediate_tif')
        self.png_save_dir = os.path.join(output_dir, 'intermediate_png') # For PNG comparison
        os.makedirs(self.tif_save_dir, exist_ok=True)
        os.makedirs(self.png_save_dir, exist_ok=True)

        # Get a fixed batch from validation set for consistent saving
        self.fixed_val_batch = None
        if self.val_ds:
             # Use tf.data.Dataset methods for potentially large datasets
             self.fixed_val_batch = next(iter(self.val_ds.take(1)))
        if self.fixed_val_batch is None:
             print("Warning: No validation data found for intermediate image saving.")

    def on_epoch_end(self, epoch, logs=None):
        # Check if it's time to save based on frequency AND if we have data
        if self.fixed_val_batch and (epoch + 1) % self.save_freq == 0:
            lr_img, hr_img = self.fixed_val_batch # These are [0, 255] float32, shape (1, H, W, C)

            # Generate the SR image tensor ([0, 255] float32), shape (1, H, W, C)
            # Run generator in inference mode
            sr_tensor = self.generator(lr_img, training=False)

            # --- Diagnostics on the generated SR tensor ---
            try:
                # Get the single image from the batch and convert to NumPy
                sr_array = sr_tensor[0].numpy() # Shape (H, W, C)
                lr_array = lr_img[0].numpy()     # Shape (H_lr, W_lr, C)
                hr_array = hr_img[0].numpy()     # Shape (H, W, C)

                print("\n--- SR Array Diagnostics (Before Saving) ---")
                print(f"Epoch: {epoch+1}")
                print(f"SR Shape: {sr_array.shape}, LR Shape: {lr_array.shape}, HR Shape: {hr_array.shape}")
                print(f"SR dtype: {sr_array.dtype}")
                print(f"SR Min Value: {np.nanmin(sr_array):.4f}")
                print(f"SR Max Value: {np.nanmax(sr_array):.4f}")
                print(f"SR Mean Value: {np.nanmean(sr_array):.4f}")
                has_nan = np.isnan(sr_array).any()
                has_inf = np.isinf(sr_array).any()
                print(f"SR Contains NaN: {has_nan}")
                print(f"SR Contains Inf: {has_inf}")
                print("--------------------------------------------\n")

                # Basic sanity check before attempting to save
                if has_nan or has_inf:
                     print("WARNING: SR Array contains NaN or Inf values. Skipping image saving for this epoch.")
                     return
                if not (sr_array.ndim == 3 and sr_array.shape[-1] == 3):
                    print(f"WARNING: SR array shape {sr_array.shape} is not HWC. Skipping image saving.")
                    return

            except Exception as e:
                print(f"Error during image diagnostics: {e}")
                return # Stop if diagnostics fail

            # --- Save SR as Float32 TIF (for analysis) ---
            try:
                # Ensure float32 dtype (should be, but verify)
                if sr_array.dtype != np.float32:
                    sr_array_f32 = sr_array.astype(np.float32)
                else:
                    sr_array_f32 = sr_array

                save_path_f32 = os.path.join(self.tif_save_dir, f'epoch_{(epoch+1):04d}_SR_float32.tif')
                tifffile.imwrite(save_path_f32, sr_array_f32, imagej=False, metadata={'axes': 'YXC'})
                # print(f"Saved SR float32 TIF to {save_path_f32}") # Can be verbose
            except Exception as e_f32:
                print(f"Error saving SR float32 TIF with tifffile: {e_f32}")

            # --- Save SR as uint8 PNG (for visual debugging) ---
            try:
                # Clip values to [0, 255] and cast to uint8
                sr_image_clipped = np.clip(sr_array, 0, 255)
                sr_image_uint8 = sr_image_clipped.astype(np.uint8)

                save_path_png = os.path.join(self.png_save_dir, f'epoch_{(epoch+1):04d}_SR_uint8.png')
                # Use PIL for straightforward saving
                img_pil = Image.fromarray(sr_image_uint8)
                img_pil.save(save_path_png)
                # print(f"Saved SR uint8 PNG to {save_path_png}") # Can be verbose
            except Exception as e_png:
                print(f"Error saving SR uint8 PNG: {e_png}")

            # --- Optionally save LR/HR as uint8 PNG ---
            try:
                lr_path_png = os.path.join(self.png_save_dir, f'epoch_{(epoch+1):04d}_LR_uint8.png')
                hr_path_png = os.path.join(self.png_save_dir, f'epoch_{(epoch+1):04d}_HR_uint8.png')

                Image.fromarray(np.clip(lr_array, 0, 255).astype(np.uint8)).save(lr_path_png)
                Image.fromarray(np.clip(hr_array, 0, 255).astype(np.uint8)).save(hr_path_png)
                print(f"Saved SR/LR/HR uint8 PNGs and SR float32 TIF for epoch {epoch+1}")
            except Exception as e_lh_png:
                 print(f"Error saving LR/HR uint8 PNGs: {e_lh_png}")

# --- Main Training Script ---

def main(args):
    # --- Setup ---
    # Create output directories
    ckpt_dir = os.path.join(args.outputdir, 'checkpoints')
    img_save_dir = args.outputdir # Base dir for SaveIntermediateImages callback
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Load Data ---
    print("Loading datasets...")
    train_ds, val_ds = get_oli2msi_datasets(args.datadir, args.batch_size, args.val_split)
    
    if train_ds is None:
        print("Error loading training dataset. Exiting.")
        return
        
    # Calculate steps per epoch
    # This requires knowing the dataset size, which tf.data doesn't easily provide beforehand
    # Estimate based on total pairs and split (less accurate if filtering happened)
    # Or iterate once (can be slow)
    # Let's use tf.data.experimental.cardinality
    train_cardinality = tf.data.experimental.cardinality(train_ds).numpy()
    if train_cardinality == tf.data.experimental.UNKNOWN_CARDINALITY or train_cardinality == tf.data.experimental.INFINITE_CARDINALITY:
        print("Warning: Could not determine training dataset size. Steps per epoch might be inaccurate if dataset size is large.")
        # Fallback: Calculate based on initial file count (less accurate due to filtering/split)
        # total_pairs = 1000 # As per initial description 
        # approx_train_size = int(total_pairs * (1 - args.val_split))
        # steps_per_epoch = max(1, approx_train_size // args.batch_size)
        # Or set a fixed large number if cardinality unknown
        steps_per_epoch = 500 # Default if cardinality unknown
    else:
        steps_per_epoch = train_cardinality
    print(f"Steps per epoch: {steps_per_epoch}")

    val_cardinality = tf.data.experimental.cardinality(val_ds).numpy() if val_ds else 0

    # --- Build Models ---
    print("Building models...")
    generator = create_generator() # Uses the modified sr_resnet from model.srgan
    discriminator = create_discriminator() # Uses the modified discriminator

    # Build VGG model for perceptual loss
    vgg = srgan._vgg(CONTENT_LOSS_LAYER)
    vgg.trainable = False

    # --- Optimizers and Losses ---
    print("Setting up optimizers and losses...")
    g_optimizer = Adam(learning_rate=args.g_lr)
    d_optimizer = Adam(learning_rate=args.d_lr)

    mse = MeanSquaredError()

    # --- Checkpointing ---
    g_ckpt = tf.train.Checkpoint(optimizer=g_optimizer, model=generator)
    d_ckpt = tf.train.Checkpoint(optimizer=d_optimizer, model=discriminator)
    
    g_ckpt_manager = tf.train.CheckpointManager(g_ckpt, os.path.join(ckpt_dir, 'generator'), max_to_keep=3)
    d_ckpt_manager = tf.train.CheckpointManager(d_ckpt, os.path.join(ckpt_dir, 'discriminator'), max_to_keep=3)

    # Restore latest checkpoint if exists
    if g_ckpt_manager.latest_checkpoint:
        g_ckpt.restore(g_ckpt_manager.latest_checkpoint)
        print(f"Restored generator from {g_ckpt_manager.latest_checkpoint}")
    if d_ckpt_manager.latest_checkpoint:
        d_ckpt.restore(d_ckpt_manager.latest_checkpoint)
        print(f"Restored discriminator from {d_ckpt_manager.latest_checkpoint}")
        
    initial_epoch = g_ckpt.optimizer.iterations.numpy() // steps_per_epoch # Estimate starting epoch

    # --- Define Training Step ---
    @tf.function
    def train_step(lr, hr, generator, discriminator, vgg, mse, g_optimizer, d_optimizer):
        # lr, hr are [0, 255] float32.
        # generator output sr is also [0, 255] float32.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate SR image
            sr = generator(lr, training=True)

            # Normalize inputs for models/losses where needed
            hr_norm_m11 = normalize_m11(hr) # For discriminator [-1, 1]
            sr_norm_m11 = normalize_m11(sr) # For discriminator [-1, 1]
            hr_vgg = preprocess_input(hr)   # For VGG loss
            sr_vgg = preprocess_input(sr)   # For VGG loss

            # Discriminator outputs (logits for LSGAN)
            hr_output = discriminator(hr_norm_m11, training=True)
            sr_output = discriminator(sr_norm_m11, training=True)

            # --- Calculate Losses ---
            # 1. Content Loss (VGG MSE)
            hr_vgg_features = vgg(hr_vgg)
            sr_vgg_features = vgg(sr_vgg)
            content_loss = mse(hr_vgg_features, sr_vgg_features)
            
            # 2. L1 Pixel Loss (on [0, 255] scale)
            l1_loss = tf.reduce_mean(tf.abs(hr - sr))

            # 3. Adversarial Loss (LSGAN - MSE)
            # Generator wants discriminator to label fake images as real (label=1)
            gen_adversarial_loss = mse(tf.ones_like(sr_output), sr_output)

            # 4. Discriminator Loss (LSGAN - MSE)
            # Label real images as 1, fake images as 0
            hr_loss = mse(tf.ones_like(hr_output), hr_output)
            sr_loss = mse(tf.zeros_like(sr_output), sr_output)
            # Original LSGAN paper uses 0.5 * (hr_loss + sr_loss)
            total_disc_loss = 0.5 * (hr_loss + sr_loss)
            # --- End Loss Calculation ---

            # Combined Generator Loss (Weighted Sum)
            # Weights defined outside: L1_WEIGHT, CONTENT_WEIGHT, ADVERSARIAL_WEIGHT
            total_gen_loss = (L1_WEIGHT * l1_loss) + \
                             (CONTENT_WEIGHT * content_loss) + \
                             (ADVERSARIAL_WEIGHT * gen_adversarial_loss)

        # Calculate and apply gradients
        g_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        d_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

        # Apply gradients if they exist (avoid issues if a model has no trainable vars)
        if g_gradients:
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        if d_gradients:
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # Return all individual losses for tracking
        return total_gen_loss, total_disc_loss, l1_loss, content_loss, gen_adversarial_loss

    # --- Define Validation Step ---
    @tf.function
    def validation_step(lr, hr, generator):
        # lr and hr are float32 tensors in the range [0, 255]
        sr = generator(lr, training=False)

        # --- Calculate PSNR ---
        hr_uint8 = tf.cast(tf.clip_by_value(hr, 0, 255), tf.uint8)
        sr_uint8 = tf.cast(tf.clip_by_value(sr, 0, 255), tf.uint8)
        current_psnr = psnr(hr_uint8, sr_uint8)
        # --- End PSNR Calculation ---

        return current_psnr

    # --- Callbacks ---
    print("Setting up callbacks...")
    callbacks = []

    # 1. Image Saving Callback
    if val_ds:
        image_saver = SaveIntermediateImages(generator, val_ds, img_save_dir, save_freq=args.save_freq)
        callbacks.append(image_saver)
    else:
        print("Skipping intermediate image saving as no validation dataset is available.")

    # 2. Early Stopping Callback
    early_stopping = EarlyStopping(
        monitor='val_psnr',
        patience=args.patience,
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    early_stopping.set_model(generator) # Assign model reference needed for state
    callbacks.append(early_stopping)

    # Optional: Reduce Learning Rate on Plateau
    # reduce_lr = ReduceLROnPlateau(monitor='val_psnr', factor=0.5, patience=5, mode='max', verbose=1)
    # reduce_lr.set_model(generator)
    # callbacks.append(reduce_lr)

    # --- Call on_train_begin for all callbacks ---
    print("Calling on_train_begin for callbacks...")
    for callback in callbacks:
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin()
    # --- End on_train_begin calls ---

    # 3. Checkpoint Saving (Managed within the loop based on best val_psnr)
    best_val_psnr = -np.inf # Initialize best validation PSNR
    if g_ckpt_manager.latest_checkpoint and d_ckpt_manager.latest_checkpoint:
         print("Checkpoints found. Attempting to load previous best PSNR if tracked...")
         # TODO: Implement loading best_val_psnr from a file saved alongside checkpoints
         pass

    print(f"Starting training from epoch {initial_epoch + 1} for {args.epochs} epochs...")
    start_time = time.time()

    # Metrics trackers for each epoch
    gen_loss_tracker = Mean(name='generator_loss')
    disc_loss_tracker = Mean(name='discriminator_loss')
    l1_loss_tracker = Mean(name='l1_loss') # Add L1 tracker
    content_loss_tracker = Mean(name='content_loss')
    adv_loss_tracker = Mean(name='adversarial_loss')
    val_psnr_tracker = Mean(name='validation_psnr') # Use PSNR tracker

    # --- Training Loop ---
    stop_training_flag = False # Flag to break outer loop if needed
    for epoch in range(initial_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        
        # --- Call on_epoch_begin for all callbacks ---
        for callback in callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch)
        # --- End on_epoch_begin calls ---

        # Reset trackers at the start of each epoch
        gen_loss_tracker.reset_states()
        disc_loss_tracker.reset_states()
        l1_loss_tracker.reset_states() # Reset L1 tracker
        content_loss_tracker.reset_states()
        adv_loss_tracker.reset_states()
        val_psnr_tracker.reset_states()

        # --- Training Phase ---
        step = 0
        for lr_batch, hr_batch in train_ds:
            # --- Call on_train_batch_begin ---
            for callback in callbacks:
                if hasattr(callback, 'on_train_batch_begin'):
                    callback.on_train_batch_begin(step)
            # --- End on_train_batch_begin ---

            g_loss, d_loss, l1, content, adv = train_step(
                lr_batch, hr_batch, generator, discriminator, vgg, mse, g_optimizer, d_optimizer
            )

            # Update training metrics
            gen_loss_tracker(g_loss)
            disc_loss_tracker(d_loss)
            l1_loss_tracker(l1) # Track L1 loss
            content_loss_tracker(content)
            adv_loss_tracker(adv)
            
            if step % args.log_freq == 0:
                print(f"Step {step}/{steps_per_epoch} - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, L1: {l1:.2f}, Content: {content:.2f}, Adv: {adv:.4f}")
            step += 1

            # --- Call on_train_batch_end ---
            batch_logs = {'generator_loss': g_loss, 'discriminator_loss': d_loss} # Minimal logs for batch end
            for callback in callbacks:
                if hasattr(callback, 'on_train_batch_end'):
                    callback.on_train_batch_end(step, logs=batch_logs)
            # --- End on_train_batch_end ---

        # --- Validation Phase ---
        logs = { # Prepare logs dictionary for epoch end
            'generator_loss': gen_loss_tracker.result(),
            'discriminator_loss': disc_loss_tracker.result(),
            'l1_loss': l1_loss_tracker.result(), # Add L1 loss to logs
            'content_loss': content_loss_tracker.result(),
            'adversarial_loss': adv_loss_tracker.result()
        }

        if val_ds:
            # --- Call on_test_begin ---
            for callback in callbacks:
                if hasattr(callback, 'on_test_begin'):
                    callback.on_test_begin()
            # --- End on_test_begin ---

            val_psnr_tracker.reset_states() # Reset validation tracker
            val_step = 0
            for val_lr, val_hr in val_ds:
                # --- Call on_test_batch_begin ---
                for callback in callbacks:
                    if hasattr(callback, 'on_test_batch_begin'):
                        callback.on_test_batch_begin(val_step)
                # --- End on_test_batch_begin ---

                v_psnr = validation_step(val_lr, val_hr, generator)
                if tf.rank(v_psnr) > 0:
                    v_psnr = tf.reduce_mean(v_psnr)
                val_psnr_tracker(v_psnr)

                # --- Call on_test_batch_end ---
                val_batch_logs = {'val_psnr_batch': v_psnr} # Minimal logs for val batch end
                for callback in callbacks:
                    if hasattr(callback, 'on_test_batch_end'):
                        callback.on_test_batch_end(val_step, logs=val_batch_logs)
                # --- End on_test_batch_end ---
                val_step += 1

            current_val_psnr = val_psnr_tracker.result().numpy()
            logs['val_psnr'] = current_val_psnr # Add val_psnr to logs

            # --- Call on_test_end ---
            for callback in callbacks:
                if hasattr(callback, 'on_test_end'):
                    callback.on_test_end()
            # --- End on_test_end ---

            print(f"Epoch {epoch+1} - Avg Train G Loss: {gen_loss_tracker.result():.4f}, Avg Train D Loss: {disc_loss_tracker.result():.4f}, Avg L1: {l1_loss_tracker.result():.2f}")
            print(f"Epoch {epoch+1} - Avg Validation PSNR: {current_val_psnr:.4f}")

            # --- Checkpoint Saving Logic (Save if validation PSNR improved) ---
            if current_val_psnr > best_val_psnr:
                print(f"Validation PSNR improved from {best_val_psnr:.4f} to {current_val_psnr:.4f}. Saving checkpoints.")
                best_val_psnr = current_val_psnr
                g_ckpt_manager.save()
                d_ckpt_manager.save()
                # TODO: Optionally save best_val_psnr to a file here
            else:
                print(f"Validation PSNR ({current_val_psnr:.4f}) did not improve from best ({best_val_psnr:.4f}).")

            # --- Callback Checks and on_epoch_end Calls ---
            early_stopping.model.stop_training = False # Reset stop flag before checks
            for callback in callbacks:
                 if hasattr(callback, 'on_epoch_end'):
                    print(f"Calling on_epoch_end for {type(callback).__name__} with logs: {logs}") # Debug print
                    callback.on_epoch_end(epoch, logs=logs) # Pass logs dict

            # Check the EarlyStopping callback's flag *after* on_epoch_end has been called
            if early_stopping.model.stop_training:
                print(f"Early stopping triggered after epoch {epoch + 1}.")
                stop_training_flag = True # Set flag to break outer loop

        else: # No validation data
            print(f"Epoch {epoch+1} - Avg Train G Loss: {gen_loss_tracker.result():.4f}, Avg Train D Loss: {disc_loss_tracker.result():.4f}")
            print("No validation data. Saving checkpoints every epoch.")
            g_ckpt_manager.save()
            d_ckpt_manager.save()
            # Run only image saving callback if it exists (but it depends on val_ds, so likely won't run)
            for callback in callbacks:
                 if isinstance(callback, SaveIntermediateImages):
                     callback.on_epoch_end(epoch, logs=logs) # Pass logs

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

        if stop_training_flag:
            break # Exit the main training loop

    # --- Call on_train_end for all callbacks ---
    print("\nTraining finished. Calling on_train_end for callbacks...")
    for callback in callbacks:
        if hasattr(callback, 'on_train_end'):
            callback.on_train_end()
    # --- End on_train_end calls ---

    total_time = time.time() - start_time
    print(f"Total training duration: {total_time / 60:.2f} minutes.")

    # --- Save Final Weights as .h5 ---
    print("\nSaving final model weights in .h5 format...")
    final_gen_weights_path = os.path.join(args.outputdir, 'final_generator_weights.h5')
    final_disc_weights_path = os.path.join(args.outputdir, 'final_discriminator_weights.h5')

    try:
        generator.save_weights(final_gen_weights_path)
        print(f"Saved final generator weights to: {final_gen_weights_path}")
        discriminator.save_weights(final_disc_weights_path)
        print(f"Saved final discriminator weights to: {final_disc_weights_path}")
        
        # Optional Note:
        if any(isinstance(cb, EarlyStopping) and cb.restore_best_weights for cb in callbacks):
             print("(Note: Saved weights should be the 'best' according to EarlyStopping monitor)")
        else:
             print("(Note: Saved weights are from the final training epoch)")
             
    except Exception as e:
        print(f"Error saving final .h5 weights: {e}")
    # --- End H5 Saving ---

    # --- Post-Training Evaluation ---
    if early_stopping.stopped_epoch > 0: # Check if early stopping actually stopped the training
         print(f"\nTraining stopped early at epoch {early_stopping.stopped_epoch + 1}.") # Added newline

    # If early stopping restored weights, the best weights are already in the models.
    # Load the best weights explicitly if restore_best_weights=False was used
    if not early_stopping.restore_best_weights and best_val_psnr > -np.inf:
        print("Restoring best weights based on tracked validation PSNR...")
        # Need to figure out which checkpoint corresponds to best_val_psnr
        # This requires more complex checkpoint management (e.g., naming by epoch/metric)
        # Or rely on CheckpointManager's latest if saving only happened on improvement.
        if g_ckpt_manager.latest_checkpoint and d_ckpt_manager.latest_checkpoint:
             print("Restoring latest checkpoint (assumed best based on saving logic).")
             g_ckpt.restore(g_ckpt_manager.latest_checkpoint).expect_partial() # Use expect_partial if optimizer state changes
             d_ckpt.restore(d_ckpt_manager.latest_checkpoint).expect_partial()
        else:
             print("Could not restore best weights (no checkpoints found or tracking logic needed).")


    # Optional: Final evaluation on validation set using best weights
    if val_ds:
        print("\nEvaluating final model performance on validation set...")
        if early_stopping.restore_best_weights:
             print("(Using best weights restored by EarlyStopping)")
        else:
             print("(Using weights from last epoch or manually restored best)")

        final_val_psnr_tracker = Mean(name='final_validation_psnr')
        for val_lr, val_hr in val_ds:
            v_psnr = validation_step(val_lr, val_hr, generator)
            if tf.rank(v_psnr) > 0: v_psnr = tf.reduce_mean(v_psnr)
            final_val_psnr_tracker(v_psnr)
        final_psnr = final_val_psnr_tracker.result().numpy()
        print(f"Final Average Validation PSNR: {final_psnr:.4f}")
        if best_val_psnr > -np.inf:
            print(f"(Best PSNR achieved during training: {best_val_psnr:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SRGAN on OLI2MSI dataset.") # Add description
    parser.add_argument('-d', '--datadir', type=str, required=True, help='Directory containing train_lr/ and train_hr/ subfolders with .TIF files')
    parser.add_argument('-o', '--outputdir', type=str, default='./output', help='Directory to save checkpoints and intermediate images')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for training') # Reduced default
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of data to use for validation') # Adjusted default
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (epochs)') # Added Arg
    parser.add_argument('--log_freq', type=int, default=50, help='Log training loss every N steps')
    parser.add_argument('--save_freq', type=int, default=5, help='Save intermediate images every N epochs') # Added Arg

    args = parser.parse_args()
    
    print("--- SRGAN Training Start ---")
    print(f"Dataset: {args.datadir}")
    print(f"Output: {args.outputdir}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"Gen LR: {args.g_lr}, Disc LR: {args.d_lr}")
    print(f"Early Stopping Patience: {args.patience} (monitoring val_psnr, mode=max)") # Updated print
    print(f"Intermediate Image Save Freq: {args.save_freq} epochs") # Added print
    print("\nWARNING: Using Binary Crossentropy (BCE) for the discriminator may lead to the discriminator loss stalling near zero, especially early in training. Monitor the 'Avg Train D Loss' value. If it consistently stays very close to zero while the generator loss remains high or validation PSNR does not improve, consider potential adjustments like significantly lowering the discriminator learning rate (--d_lr), using label smoothing, or switching to a different GAN loss formulation (e.g., Wasserstein loss) in future experiments.\n") # Added Warning

    main(args)
