import os
import matplotlib
matplotlib.use('Agg')  
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio 
import tifffile 

from PIL import Image 
from model import srgan
from model.srgan import generator as create_generator
from model.srgan import discriminator as create_discriminator
from data import get_oli2msi_datasets
from model.common import normalize_m11, psnr 
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.metrics import Mean

# VGG54 (original) uses layer 20 (block5_conv4)
CONTENT_LOSS_LAYER = 10 


L1_WEIGHT = 1e-2
CONTENT_WEIGHT = 6e-3 # Chosen relative to adversarial weight (1e-3)
ADVERSARIAL_WEIGHT = 1e-3

class SaveIntermediateImages(Callback):
    def __init__(self, generator_model, validation_dataset, output_dir, save_freq=5): 
        super().__init__()
        self.generator = generator_model
        self.val_ds = validation_dataset 
        self.output_dir = output_dir
        self.save_freq = save_freq
        self.tif_save_dir = os.path.join(output_dir, 'intermediate_tif')
        self.png_save_dir = os.path.join(output_dir, 'intermediate_png') 
        os.makedirs(self.tif_save_dir, exist_ok=True)
        os.makedirs(self.png_save_dir, exist_ok=True)

        self.fixed_val_batch = None
        if self.val_ds:
             self.fixed_val_batch = next(iter(self.val_ds.take(1)))
        if self.fixed_val_batch is None:
             print("Warning: No validation data found for intermediate image saving.")

    def on_epoch_end(self, epoch, logs=None):
        if self.fixed_val_batch and (epoch + 1) % self.save_freq == 0:
            lr_img, hr_img = self.fixed_val_batch 
            sr_tensor = self.generator(lr_img, training=False)
            try:
                sr_array = sr_tensor[0].numpy() 
                lr_array = lr_img[0].numpy()     
                hr_array = hr_img[0].numpy()     
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


                if has_nan or has_inf:
                     print("WARNING: SR Array contains NaN or Inf values. Skipping image saving for this epoch.")
                     return
                if not (sr_array.ndim == 3 and sr_array.shape[-1] == 3):
                    print(f"WARNING: SR array shape {sr_array.shape} is not HWC. Skipping image saving.")
                    return

            except Exception as e:
                print(f"Error during image diagnostics: {e}")
                return

            try:
                if sr_array.dtype != np.float32:
                    sr_array_f32 = sr_array.astype(np.float32)
                else:
                    sr_array_f32 = sr_array

                save_path_f32 = os.path.join(self.tif_save_dir, f'epoch_{(epoch+1):04d}_SR_float32.tif')
                tifffile.imwrite(save_path_f32, sr_array_f32, imagej=False, metadata={'axes': 'YXC'})
            except Exception as e_f32:
                print(f"Error saving SR float32 TIF with tifffile: {e_f32}")

            try:
                sr_image_clipped = np.clip(sr_array, 0, 255)
                sr_image_uint8 = sr_image_clipped.astype(np.uint8)

                save_path_png = os.path.join(self.png_save_dir, f'epoch_{(epoch+1):04d}_SR_uint8.png')
                img_pil = Image.fromarray(sr_image_uint8)
                img_pil.save(save_path_png)
            except Exception as e_png:
                print(f"Error saving SR uint8 PNG: {e_png}")

            try:
                lr_path_png = os.path.join(self.png_save_dir, f'epoch_{(epoch+1):04d}_LR_uint8.png')
                hr_path_png = os.path.join(self.png_save_dir, f'epoch_{(epoch+1):04d}_HR_uint8.png')

                Image.fromarray(np.clip(lr_array, 0, 255).astype(np.uint8)).save(lr_path_png)
                Image.fromarray(np.clip(hr_array, 0, 255).astype(np.uint8)).save(hr_path_png)
                print(f"Saved SR/LR/HR uint8 PNGs and SR float32 TIF for epoch {epoch+1}")
            except Exception as e_lh_png:
                 print(f"Error saving LR/HR uint8 PNGs: {e_lh_png}")

def main(args):
    ckpt_dir = os.path.join(args.outputdir, 'checkpoints')
    img_save_dir = args.outputdir 
    os.makedirs(ckpt_dir, exist_ok=True)
    print("Loading datasets...")
    train_ds, val_ds = get_oli2msi_datasets(args.datadir, args.batch_size, args.val_split)
    
    if train_ds is None:
        print("Error loading training dataset. Exiting.")
        return
        
    train_cardinality = tf.data.experimental.cardinality(train_ds).numpy()
    if train_cardinality == tf.data.experimental.UNKNOWN_CARDINALITY or train_cardinality == tf.data.experimental.INFINITE_CARDINALITY:
        print("Warning: Could not determine training dataset size. Steps per epoch might be inaccurate if dataset size is large.")
        steps_per_epoch = 500 
    else:
        steps_per_epoch = train_cardinality
    print(f"Steps per epoch: {steps_per_epoch}")

    val_cardinality = tf.data.experimental.cardinality(val_ds).numpy() if val_ds else 0


    print("Building models...")
    generator = create_generator() 
    discriminator = create_discriminator() 

    vgg = srgan._vgg(CONTENT_LOSS_LAYER)
    vgg.trainable = False

    print("Setting up optimizers and losses...")
    g_optimizer = Adam(learning_rate=args.g_lr)
    d_optimizer = Adam(learning_rate=args.d_lr)

    mse = MeanSquaredError()

    g_ckpt = tf.train.Checkpoint(optimizer=g_optimizer, model=generator)
    d_ckpt = tf.train.Checkpoint(optimizer=d_optimizer, model=discriminator)
    
    g_ckpt_manager = tf.train.CheckpointManager(g_ckpt, os.path.join(ckpt_dir, 'generator'), max_to_keep=3)
    d_ckpt_manager = tf.train.CheckpointManager(d_ckpt, os.path.join(ckpt_dir, 'discriminator'), max_to_keep=3)

    if g_ckpt_manager.latest_checkpoint:
        g_ckpt.restore(g_ckpt_manager.latest_checkpoint)
        print(f"Restored generator from {g_ckpt_manager.latest_checkpoint}")
    if d_ckpt_manager.latest_checkpoint:
        d_ckpt.restore(d_ckpt_manager.latest_checkpoint)
        print(f"Restored discriminator from {d_ckpt_manager.latest_checkpoint}")
        
    initial_epoch = g_ckpt.optimizer.iterations.numpy() // steps_per_epoch 

    @tf.function
    def train_step(lr, hr, generator, discriminator, vgg, mse, g_optimizer, d_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            sr = generator(lr, training=True)
            hr_norm_m11 = normalize_m11(hr) # For discriminator [-1, 1]
            sr_norm_m11 = normalize_m11(sr) # For discriminator [-1, 1]
            hr_vgg = preprocess_input(hr)   # For VGG loss
            sr_vgg = preprocess_input(sr)   # For VGG loss

            hr_output = discriminator(hr_norm_m11, training=True)
            sr_output = discriminator(sr_norm_m11, training=True)
            hr_vgg_features = vgg(hr_vgg)
            sr_vgg_features = vgg(sr_vgg)
            content_loss = mse(hr_vgg_features, sr_vgg_features)
            l1_loss = tf.reduce_mean(tf.abs(hr - sr))
            gen_adversarial_loss = mse(tf.ones_like(sr_output), sr_output)
            hr_loss = mse(tf.ones_like(hr_output), hr_output)
            sr_loss = mse(tf.zeros_like(sr_output), sr_output)
            total_disc_loss = 0.5 * (hr_loss + sr_loss)
            total_gen_loss = (L1_WEIGHT * l1_loss) + \
                             (CONTENT_WEIGHT * content_loss) + \
                             (ADVERSARIAL_WEIGHT * gen_adversarial_loss)

        g_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        d_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

        if g_gradients:
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        if d_gradients:
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        return total_gen_loss, total_disc_loss, l1_loss, content_loss, gen_adversarial_loss

    @tf.function
    def validation_step(lr, hr, generator):
        sr = generator(lr, training=False)
        hr_uint8 = tf.cast(tf.clip_by_value(hr, 0, 255), tf.uint8)
        sr_uint8 = tf.cast(tf.clip_by_value(sr, 0, 255), tf.uint8)
        current_psnr = psnr(hr_uint8, sr_uint8)
        return current_psnr

    print("Setting up callbacks...")
    callbacks = []

    if val_ds:
        image_saver = SaveIntermediateImages(generator, val_ds, img_save_dir, save_freq=args.save_freq)
        callbacks.append(image_saver)
    else:
        print("Skipping intermediate image saving as no validation dataset is available.")

    early_stopping = EarlyStopping(
        monitor='val_psnr',
        patience=args.patience,
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    early_stopping.set_model(generator) 
    callbacks.append(early_stopping)
    print("Calling on_train_begin for callbacks...")
    for callback in callbacks:
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin()

    best_val_psnr = -np.inf 
    if g_ckpt_manager.latest_checkpoint and d_ckpt_manager.latest_checkpoint:
         print("Checkpoints found. Attempting to load previous best PSNR if tracked...")
         pass

    print(f"Starting training from epoch {initial_epoch + 1} for {args.epochs} epochs...")
    start_time = time.time()

    gen_loss_tracker = Mean(name='generator_loss')
    disc_loss_tracker = Mean(name='discriminator_loss')
    l1_loss_tracker = Mean(name='l1_loss') # Add L1 tracker
    content_loss_tracker = Mean(name='content_loss')
    adv_loss_tracker = Mean(name='adversarial_loss')
    val_psnr_tracker = Mean(name='validation_psnr') # Use PSNR tracker

    stop_training_flag = False 
    for epoch in range(initial_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        
        for callback in callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch)

        gen_loss_tracker.reset_states()
        disc_loss_tracker.reset_states()
        l1_loss_tracker.reset_states() # Reset L1 tracker
        content_loss_tracker.reset_states()
        adv_loss_tracker.reset_states()
        val_psnr_tracker.reset_states()

        #  Training Phase 
        step = 0
        for lr_batch, hr_batch in train_ds:
            for callback in callbacks:
                if hasattr(callback, 'on_train_batch_begin'):
                    callback.on_train_batch_begin(step)

            g_loss, d_loss, l1, content, adv = train_step(
                lr_batch, hr_batch, generator, discriminator, vgg, mse, g_optimizer, d_optimizer
            )

            gen_loss_tracker(g_loss)
            disc_loss_tracker(d_loss)
            l1_loss_tracker(l1) # Track L1 loss
            content_loss_tracker(content)
            adv_loss_tracker(adv)
            
            if step % args.log_freq == 0:
                print(f"Step {step}/{steps_per_epoch} - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, L1: {l1:.2f}, Content: {content:.2f}, Adv: {adv:.4f}")
            step += 1

            batch_logs = {'generator_loss': g_loss, 'discriminator_loss': d_loss} 
            for callback in callbacks:
                if hasattr(callback, 'on_train_batch_end'):
                    callback.on_train_batch_end(step, logs=batch_logs)


        #  Validation Phase 
        logs = { 
            'generator_loss': gen_loss_tracker.result(),
            'discriminator_loss': disc_loss_tracker.result(),
            'l1_loss': l1_loss_tracker.result(),
            'content_loss': content_loss_tracker.result(),
            'adversarial_loss': adv_loss_tracker.result()
        }

        if val_ds:
            for callback in callbacks:
                if hasattr(callback, 'on_test_begin'):
                    callback.on_test_begin()

            val_psnr_tracker.reset_states()
            val_step = 0
            for val_lr, val_hr in val_ds:
                for callback in callbacks:
                    if hasattr(callback, 'on_test_batch_begin'):
                        callback.on_test_batch_begin(val_step)

                v_psnr = validation_step(val_lr, val_hr, generator)
                if tf.rank(v_psnr) > 0:
                    v_psnr = tf.reduce_mean(v_psnr)
                val_psnr_tracker(v_psnr)

                val_batch_logs = {'val_psnr_batch': v_psnr} 
                for callback in callbacks:
                    if hasattr(callback, 'on_test_batch_end'):
                        callback.on_test_batch_end(val_step, logs=val_batch_logs)
                val_step += 1

            current_val_psnr = val_psnr_tracker.result().numpy()
            logs['val_psnr'] = current_val_psnr

            for callback in callbacks:
                if hasattr(callback, 'on_test_end'):
                    callback.on_test_end()

            print(f"Epoch {epoch+1} - Avg Train G Loss: {gen_loss_tracker.result():.4f}, Avg Train D Loss: {disc_loss_tracker.result():.4f}, Avg L1: {l1_loss_tracker.result():.2f}")
            print(f"Epoch {epoch+1} - Avg Validation PSNR: {current_val_psnr:.4f}")

            if current_val_psnr > best_val_psnr:
                print(f"Validation PSNR improved from {best_val_psnr:.4f} to {current_val_psnr:.4f}. Saving checkpoints.")
                best_val_psnr = current_val_psnr
                g_ckpt_manager.save()
                d_ckpt_manager.save()
            else:
                print(f"Validation PSNR ({current_val_psnr:.4f}) did not improve from best ({best_val_psnr:.4f}).")

            early_stopping.model.stop_training = False
            for callback in callbacks:
                 if hasattr(callback, 'on_epoch_end'):
                    print(f"Calling on_epoch_end for {type(callback).__name__} with logs: {logs}") # Debug print
                    callback.on_epoch_end(epoch, logs=logs) 

            if early_stopping.model.stop_training:
                print(f"Early stopping triggered after epoch {epoch + 1}.")
                stop_training_flag = True

        else: 
            print(f"Epoch {epoch+1} - Avg Train G Loss: {gen_loss_tracker.result():.4f}, Avg Train D Loss: {disc_loss_tracker.result():.4f}")
            print("No validation data. Saving checkpoints every epoch.")
            g_ckpt_manager.save()
            d_ckpt_manager.save()

            for callback in callbacks:
                 if isinstance(callback, SaveIntermediateImages):
                     callback.on_epoch_end(epoch, logs=logs) # Pass logs

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

        if stop_training_flag:
            break


    print("\nTraining finished. Calling on_train_end for callbacks...")
    for callback in callbacks:
        if hasattr(callback, 'on_train_end'):
            callback.on_train_end()

    total_time = time.time() - start_time
    print(f"Total training duration: {total_time / 60:.2f} minutes.")

    print("\nSaving final model weights in .h5 format...")
    final_gen_weights_path = os.path.join(args.outputdir, 'final_generator_weights.h5')
    final_disc_weights_path = os.path.join(args.outputdir, 'final_discriminator_weights.h5')

    try:
        generator.save_weights(final_gen_weights_path)
        print(f"Saved final generator weights to: {final_gen_weights_path}")
        discriminator.save_weights(final_disc_weights_path)
        print(f"Saved final discriminator weights to: {final_disc_weights_path}")
        
        if any(isinstance(cb, EarlyStopping) and cb.restore_best_weights for cb in callbacks):
             print("(Note: Saved weights should be the 'best' according to EarlyStopping monitor)")
        else:
             print("(Note: Saved weights are from the final training epoch)")
             
    except Exception as e:
        print(f"Error saving final .h5 weights: {e}")
    if early_stopping.stopped_epoch > 0: 
         print(f"\nTraining stopped early at epoch {early_stopping.stopped_epoch + 1}.") 

    if not early_stopping.restore_best_weights and best_val_psnr > -np.inf:
        print("Restoring best weights based on tracked validation PSNR...")
        if g_ckpt_manager.latest_checkpoint and d_ckpt_manager.latest_checkpoint:
             print("Restoring latest checkpoint (assumed best based on saving logic).")
             g_ckpt.restore(g_ckpt_manager.latest_checkpoint).expect_partial() 
             d_ckpt.restore(d_ckpt_manager.latest_checkpoint).expect_partial()
        else:
             print("Could not restore best weights (no checkpoints found or tracking logic needed).")


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
    parser = argparse.ArgumentParser(description="Train SRGAN on OLI2MSI dataset.") 
    parser.add_argument('-d', '--datadir', type=str, required=True)
    parser.add_argument('-o', '--outputdir', type=str, default='./output')
    parser.add_argument('-b', '--batch_size', type=int, default=8) #
    parser.add_argument('--val_split', type=float, default=0.15) 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15) 
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=5) 
    args = parser.parse_args()
    
    print("SRGAN Training Start")
    print(f"Dataset: {args.datadir}")
    print(f"Output: {args.outputdir}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"Gen LR: {args.g_lr}, Disc LR: {args.d_lr}")
    print(f"Early Stopping Patience: {args.patience} (monitoring val_psnr, mode=max)") 
    print(f"Intermediate Image Save Freq: {args.save_freq} epochs") 
    main(args)
