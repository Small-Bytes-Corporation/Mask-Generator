import os
import sys
import math
import shutil
import multiprocessing
from pathlib import Path
from functools import partial
from typing import Optional, Tuple, Set

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Paths and directories
input_dir = "OutputNoLines/"    # Input mask directory
output_dir = "OutputCleaned/"   # Cleaned masks output
debug_dir = "OutputDebug/"      # Debug visualizations
bg_debug_dir = "InputNoLines/"  # Optional background for debug

# Configuration
ray_nb = 100                    # Number of rays for seed search
fov = 180                       # Field of view in degrees
white_threshold = 127           # RGB threshold for black/white
min_total_pixels = 200          # Min total lane pixels
min_side_pixels = 50            # Min pixels per side
num_threads = 0                 # 0 = use all available cores

def _is_valid(width: int, height: int, x: int, y: int) -> bool:
    """Check if coordinates are within image bounds"""
    return 0 <= y < height and 0 <= x < width

def _get_pixels_on_line(width: int, height: int, x0: int, y0: int, x1: int, y1: int) -> list[Tuple[int, int]]:
    """Bresenham's line algorithm to get all pixels between two points"""
    points, dx, dy = [], x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0: return [(x0, y0)] if _is_valid(width, height, x0, y0) else []

    x_inc, y_inc = dx / steps, dy / steps
    x, y, last_coord = float(x0), float(y0), None

    for _ in range(int(steps) + 1):
        int_x, int_y = int(round(x)), int(round(y))
        coord = (int_x, int_y)
        if _is_valid(width, height, int_x, int_y) and coord != last_coord:
            points.append(coord)
            last_coord = coord
        x += x_inc
        y += y_inc
    return points

def _flood_fill(pixels: np.ndarray, seed_x: int, seed_y: int, white_threshold: int) -> Set[Tuple[int, int]]:
    """Flood fill algorithm to find connected white pixels"""
    height, width = pixels.shape
    if not _is_valid(width, height, seed_x, seed_y) or pixels[seed_y, seed_x] <= white_threshold:
        return set()

    region, queue, visited = set(), [(seed_x, seed_y)], {(seed_x, seed_y)}

    while queue:
        x, y = queue.pop(0)
        region.add((x, y))

        # 8-directional flood fill
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if _is_valid(width, height, nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                if pixels[ny, nx] > white_threshold:
                    queue.append((nx, ny))
    return region

def _is_occluded(pixels: np.ndarray, origin: Tuple[int, int], white_threshold: int, px: int, py: int) -> bool:
    """Check if pixel is occluded by another white pixel closer to origin"""
    ox, oy = origin
    ray = _get_pixels_on_line(pixels.shape[1], pixels.shape[0], px, py, ox, oy)

    if len(ray) <= 1: return False

    for i in range(1, len(ray)):
        cx, cy = ray[i]
        if (cx, cy) == origin: break
        if pixels[cy, cx] > white_threshold: return True

    return False

def _save_image(pixel_set: Set[Tuple[int, int]], width: int, height: int, directory: Path, filename: str):
    """Save set of pixels as binary image"""
    output_np = np.zeros((height, width), dtype=np.uint8)

    if pixel_set:
        try:
            cols, rows = zip(*pixel_set)
            output_np[np.array(rows), np.array(cols)] = 255
        except ValueError:
            pass

    img = Image.fromarray(output_np, mode='L')
    directory.mkdir(parents=True, exist_ok=True)
    img.save(directory / filename)

def _create_debug_overlay(image_rgb_orig: Image.Image, final_pixels: Set[Tuple[int, int]], 
                         bg_debug_dir: Optional[str], name: str, white_threshold: int) -> Image.Image:
    """Create debug visualization with original image and detected lanes"""
    width, height = image_rgb_orig.size
    debug_grey_color = np.array([75, 75, 75], dtype=np.uint8)

    # Load background if available
    background_img = None
    if bg_debug_dir:
        bg_path = Path(bg_debug_dir) / name
        try:
            background_img = Image.open(bg_path).convert('RGB')
            if background_img.size != (width, height):
                background_img = background_img.resize((width, height))
        except Exception as e:
            print(f"Error loading background image {bg_path}: {e}")
            background_img = None

    if background_img is None:
        background_img = image_rgb_orig

    # Darken white pixels in background
    bg_np = np.array(background_img)
    white_mask = (bg_np[:, :, 0] > white_threshold) & \
                 (bg_np[:, :, 1] > white_threshold) & \
                 (bg_np[:, :, 2] > white_threshold)
    bg_np[white_mask] = debug_grey_color

    # Create output with detected lanes highlighted
    output_img = Image.fromarray(bg_np)
    if final_pixels:
        for x, y in final_pixels:
            if _is_valid(width, height, x, y):
                output_img.putpixel((x, y), (255, 255, 255)) # White

    return output_img

def process_image(image_path: str) -> Optional[str]:
    """Main processing pipeline for a single image"""
    try:
        image_path = Path(image_path)
        name = image_path.name

        # Load and convert images
        image_rgb_orig = Image.open(image_path).convert('RGB')
        image_l = image_rgb_orig.convert('L')
        width, height = image_l.size
        pixels = np.array(image_l)
        center_x = width // 2
        origin = (center_x, height - 1)  # Bottom center

        # Calculate ray angles
        fov_rad = math.radians(fov)
        angle_step = fov_rad / (ray_nb - 1) if ray_nb > 1 else 0
        start_angle = math.radians(-90) - (fov_rad / 2.0)
        ray_angles = [start_angle + i * angle_step for i in range(ray_nb)]

        # Find initial candidate pixels
        left_seed, right_seed = None, None
        ox, oy = origin
        ray_nb_half = len(ray_angles) // 2

        # Find leftmost seed
        for angle in ray_angles[:ray_nb_half]:
            for x, y in _get_pixels_on_line(width, height, ox, oy,
                int(ox + height * math.cos(angle)),
                int(oy + height * math.sin(angle))):
                if (x, y) != origin and pixels[y, x] > white_threshold:
                    left_seed = (x, y)
                    break
            if left_seed: break

        # Find rightmost seed
        for angle in reversed(ray_angles[ray_nb_half:]):
            for x, y in _get_pixels_on_line(width, height, ox, oy,
                int(ox + height * math.cos(angle)),
                int(oy + height * math.sin(angle))):
                if (x, y) != origin and pixels[y, x] > white_threshold:
                    right_seed = (x, y)
                    break
            if right_seed: break

        # Flood fill from found seeds
        candidate_pixels = set()
        if left_seed: candidate_pixels.update(_flood_fill(pixels, *left_seed, white_threshold))
        if right_seed: candidate_pixels.update(_flood_fill(pixels, *right_seed, white_threshold))

        if not candidate_pixels:
            return None

        # Filter occluded pixels
        occlusion_filtered_pixels = {p for p in candidate_pixels
            if p != origin and not _is_occluded(pixels, origin, white_threshold, *p)}
        if not occlusion_filtered_pixels:
            return None

        # Apply size filters
        final_pixels = occlusion_filtered_pixels
        total_count = len(final_pixels)
        if total_count < min_total_pixels:
            return None

        # Check left/right distribution
        left_pixels = {p for p in final_pixels if p[0] < center_x}
        right_pixels = {p for p in final_pixels if p[0] >= center_x}
        if len(left_pixels) < min_side_pixels or len(right_pixels) < min_side_pixels:
            return None

        # Save results
        output_path = Path(output_dir)
        _save_image(final_pixels, width, height, output_path, name)

        # Save debug visualization
        debug_overlay = _create_debug_overlay(
            image_rgb_orig, final_pixels, bg_debug_dir, name, white_threshold)
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        debug_overlay.save(Path(debug_dir) / name)

        return str(output_path / name)
    except Exception as e:
        print(f"Error processing {Path(image_path).name}: {e}")
        return None

def main():
    """Main processing pipeline"""
    # Setup parallel processing
    workers = num_threads or multiprocessing.cpu_count()
    print(f"Using {workers} worker processes")

    # Prepare output directories
    for dir_path in [output_dir, debug_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # Get input files
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' not found")
        sys.exit(1)

    input_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith('.png') and not f.startswith('.')
    ]

    if not input_paths:
        print("No PNG images found")
        sys.exit(0)

    print(f"Processing {len(input_paths)} images...")

    # Process images in parallel
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap(process_image, input_paths, chunksize=1),
            total=len(input_paths),
            desc="Processing images",
            unit="img"
        ))
        processed_count = sum(1 for r in results if r is not None)

    print(f"Completed: {processed_count}/{len(input_paths)} images processed")

if __name__ == '__main__':
    main()
