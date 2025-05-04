import os
import random
import math
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps, ImageEnhance
import shutil
from tqdm import tqdm

# Paths and directories
input_dir = "InputLines"               # Original input images
mask_dir = "OutputLines"               # Corresponding mask images
aug_input_dir = "AugmentedInputLines"  # Output for augmented inputs
aug_mask_dir = "AugmentedOutputLines"  # Output for augmented masks

# Configuration
num_augmentations = 20  # Number of variations per image
num_workers = max(1, multiprocessing.cpu_count() - 1)

# Clear/create output directories
for dir_path in [aug_input_dir, aug_mask_dir]:
   if os.path.exists(dir_path):
       shutil.rmtree(dir_path)
   Path(dir_path).mkdir(exist_ok=True)

def calculate_fill_scale(width, height, angle):
   """Calculate required scaling to prevent blank areas after rotation"""
   if angle == 0:
       return 1.0
   angle_rad = math.radians(abs(angle))
   cos_a = math.cos(angle_rad)
   sin_a = math.sin(angle_rad)

   # Calculate required scaling for width and height
   req_scale_w = (abs(sin_a) * height + abs(cos_a) * width) / width
   req_scale_h = (abs(sin_a) * width + abs(cos_a) * height) / height

   return max(req_scale_w, req_scale_h)

def augment_image(img, angle, scale, flip, brightness=0, contrast=0):
   """Apply transformations to image while maintaining dimensions"""
   w, h = img.size
   original_mode = img.mode

   # Scale to prevent rotation artifacts
   fill_scale = calculate_fill_scale(w, h, angle)
   total_scale = scale * fill_scale

   # Resize, rotate, and crop back to original dimensions
   new_w, new_h = int(w * total_scale), int(h * total_scale)
   img = img.resize((new_w, new_h), Image.Resampling.BICUBIC
      if original_mode != 'L' else Image.Resampling.BILINEAR)

   img = img.rotate(angle, resample=Image.Resampling.BICUBIC
      if original_mode != 'L' else Image.Resampling.BILINEAR,
          expand=False, fillcolor=0)

   # Center crop
   curr_w, curr_h = img.size
   left = (curr_w - w) // 2
   top = (curr_h - h) // 2
   img = img.crop((left, top, left + w, top + h))

   if flip:
       img = ImageOps.mirror(img)

   # Color adjustments (skip for masks)
   if brightness != 0 and original_mode != 'L':
       img = ImageEnhance.Brightness(img).enhance(1.0 + brightness / 100.0)
   if contrast != 0 and original_mode != 'L':
       img = ImageEnhance.Contrast(img).enhance(1.0 + contrast / 100.0)

   return img

def process_image(img_path):
   """Process single image-mask pair to generate augmented versions"""
   filename = os.path.basename(img_path)
   img_number = Path(filename).stem
   mask_path = os.path.join(mask_dir, filename)

   if not os.path.exists(mask_path):
       return False

   try:
       img = Image.open(img_path)
       mask = Image.open(mask_path).convert('L') # Ensure mask is grayscale

       # Generate multiple augmented versions
       for i in range(1, num_augmentations + 1):
           # Random transformations
           angle = random.randint(-5, 5)
           scale = 1 + (random.randint(0, 10) / 100)
           flip = random.choice([True, False])
           brightness = random.randint(-20, 20)
           contrast = random.randint(-20, 20)

           # Apply same geometric transforms to both image and mask
           aug_img = augment_image(img.copy(), angle, scale, flip, brightness, contrast)
           aug_mask = augment_image(mask.copy(), angle, scale, flip)
           aug_mask = aug_mask.point(lambda p: 255 if p > 127 else 0) # Binarize mask

           # Save augmented pair
           aug_filename = f"{img_number}_{i}.png"
           aug_img.save(os.path.join(aug_input_dir, aug_filename))
           aug_mask.save(os.path.join(aug_mask_dir, aug_filename))

       return True
   except Exception as e:
       print(f"\nError processing {filename}: {e}")
       return False

def main():
   """Main processing pipeline"""
   # Find all image files
   image_files = []
   for ext in ['.png', '.jpg', '.jpeg']:
       image_files.extend(str(p) for p in Path(input_dir).glob(f"*{ext}"))

   print(f"Found {len(image_files)} images to process")
   print(f"Using {num_workers} worker processes")

   # Process in parallel with progress bar
   processed_count = 0
   with tqdm(total=len(image_files), desc="Augmenting images") as pbar:
       with ProcessPoolExecutor(max_workers=num_workers) as executor:
           for result in executor.map(process_image, image_files):
               processed_count += 1 if result else 0
               pbar.update(1)

   print(f"Augmented {processed_count} images into {len(os.listdir(aug_input_dir))}.")

if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\nInterrupted, exiting.")
       exit(1)
