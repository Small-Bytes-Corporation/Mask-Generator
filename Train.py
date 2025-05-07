import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from PIL import Image
from tqdm import tqdm

# Configuration
img_width, img_height = 347, 256
batch_size = 8
learning_rate = 1e-4
val_split, test_split = 0.05, 0.05
log_freq = 100
random_seed = 42
num_workers = 0

# Directories
image_dir = 'AugmentedInputLines'
mask_dir = 'AugmentedOutputLines'
best_model_path = 'best.pth'
metrics_csv_path = 'metrics.csv'

# Graceful shutdown handler
stop_training = False
def signal_handler(sig, frame):
    global stop_training
    stop_training = True
signal.signal(signal.SIGINT, signal_handler)

class TrackLimitDataset(Dataset):
    """Dataset for track limit segmentation, pairing input images with mask labels"""
    def __init__(self, image_dir: Path, mask_dir: Path, image_filenames: list[str],
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Filter images that have corresponding masks
        mask_files = {p.name for p in mask_dir.glob('*.png')}
        self.images = [fname for fname in image_filenames if fname in mask_files]
        print(f"Dataset initialized with {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Load as grayscale

            if self.image_transform:
                image = self.image_transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            mask = torch.clamp(mask, 0.0, 1.0)  # Ensure binary mask
            return image, mask
        except Exception as e:
            print(f"Error loading item {idx} ({img_name}): {e}")
            # Return placeholder on error
            return self.__getitem__(0) if idx > 0 else (
                torch.zeros(3, img_height, img_width),
                torch.zeros(1, img_height, img_width))

# U-Net functions
class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block with maxpool and double convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block with skip connections from encoder to decoder"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size differences with appropriate padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # Skip connection
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution layer to produce output mask"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder path with skip connections
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# Segmentation evaluation metrics
def dice_coefficient(preds, targets, smooth=1e-6):
    """Calculate Dice coefficient (F1 score) for segmentation evaluation"""
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, targets.ndim))
    intersection = (preds * targets).sum(dim=dims)
    union_p = preds.sum(dim=dims)
    union_t = targets.sum(dim=dims)
    dice = (2. * intersection + smooth) / (union_p + union_t + smooth)
    return dice.mean()

def iou_coefficient(preds, targets, smooth=1e-6):
    """Calculate IoU (Jaccard index) for segmentation evaluation"""
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = (targets > 0.5).float()

    dims = tuple(range(1, targets.ndim))
    intersection = (preds * targets).sum(dim=dims)
    union = (preds + targets).sum(dim=dims) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def pixel_accuracy(preds, targets):
    """Calculate pixel-wise accuracy for segmentation evaluation"""
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = (targets > 0.5).float()
    correct = (preds == targets).sum().float()
    return correct / targets.numel() if targets.numel() > 0 else 0.0

def prepare_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare and split datasets into train, validation and test loaders"""
    all_image_files = sorted([p.name for p in image_dir_path.glob('*.png')])
    if not all_image_files:
        raise FileNotFoundError(f"No PNG images found in {image_dir_path}")

    # Create reproducible random split
    indices = np.arange(len(all_image_files))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split data into train/val/test sets
    num_test = int(test_split * len(all_image_files))
    num_val = int(val_split * (len(all_image_files) - num_test))
    num_train = len(all_image_files) - num_test - num_val

    train_files = [all_image_files[i] for i in indices[:num_train]]
    val_files = [all_image_files[i] for i in indices[num_train:num_train + num_val]]
    test_files = [all_image_files[i] for i in indices[num_train + num_val:]]

    print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Define transforms for images and masks
    image_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    # Create datasets
    train_dataset = TrackLimitDataset(image_dir_path, mask_dir_path,
        train_files, image_transforms, mask_transforms)
    val_dataset = TrackLimitDataset(image_dir_path, mask_dir_path,
        val_files, image_transforms, mask_transforms)
    test_dataset = TrackLimitDataset(image_dir_path, mask_dir_path,
        test_files, image_transforms, mask_transforms)

    # Configure data loaders
    pin_memory = torch.cuda.is_available()
    effective_workers = max(1, os.cpu_count() // 2) if num_workers == 0 else num_workers

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=effective_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, metrics_file):
    """Train model for one epoch and log metrics"""
    model.train()
    metrics = {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}
    running_metrics = {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}
    batch_count = 0
    total_batches = 0
    current_lr = optimizer.param_groups[0]['lr']

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        if stop_training:
            break

        # Move data to device
        data = data.to(device)
        targets = targets.to(device).float()
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # Add channel dimension if needed

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)

        # Skip problematic batches
        if outputs.shape != targets.shape:
            print(f"Shape mismatch: output {outputs.shape}, target {targets.shape}. Skipping batch.")
            continue

        loss = criterion(outputs, targets)
        if torch.isnan(loss):
            print(f"NaN loss detected. Skipping batch.")
            continue

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            acc = pixel_accuracy(outputs, targets)
            iou = iou_coefficient(outputs, targets)
            dice = dice_coefficient(outputs, targets)

        batch_metrics = {
            "loss": loss.item(), "acc": acc.item(),
            "iou": iou.item(), "dice": dice.item()
        }

        # Update running metrics
        for key in metrics:
            metrics[key] += batch_metrics[key]
            running_metrics[key] += batch_metrics[key]
        batch_count += 1
        total_batches += 1

        # Log metrics periodically
        if (batch_idx + 1) % log_freq == 0 and batch_count > 0:
            avg_metrics = {k: v / batch_count for k, v in running_metrics.items()}
            file_exists = metrics_file.exists()

            with open(metrics_file, 'a') as f:
                if not file_exists or metrics_file.stat().st_size == 0:
                    f.write("epoch,step,train_loss,train_acc,train_iou,train_dice,lr\n")

                f.write(f"{epoch},{batch_idx+1},{avg_metrics['loss']:.6f},"
                        f"{avg_metrics['acc']:.6f},{avg_metrics['iou']:.6f},"
                        f"{avg_metrics['dice']:.6f},{current_lr:.8f}\n")

            running_metrics = {k: 0.0 for k in running_metrics}
            batch_count = 0

        progress_bar.set_postfix(**batch_metrics)

    # Calculate average metrics for the epoch
    if total_batches > 0:
        for key in metrics:
            metrics[key] /= total_batches
    return metrics

def evaluate(model, loader, criterion, device, phase="val"):
    """Evaluate model on validation or test data"""
    model.eval()
    metrics = {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}
    total_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Evaluating ({phase})")
        for data, targets in progress_bar:
            if stop_training and phase == "val":
                break

            # Move data to device
            data = data.to(device)
            targets = targets.to(device).float()
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)  # Add channel dimension if needed

            # Forward pass
            outputs = model(data)
            if outputs.shape != targets.shape:
                continue

            # Calculate loss and metrics
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                continue

            acc = pixel_accuracy(outputs, targets)
            iou = iou_coefficient(outputs, targets)
            dice = dice_coefficient(outputs, targets)

            metrics["loss"] += loss.item()
            metrics["acc"] += acc.item()
            metrics["iou"] += iou.item()
            metrics["dice"] += dice.item()
            total_batches += 1

            batch_metrics = {
                "loss": loss.item(), "acc": acc.item(),
                "iou": iou.item(), "dice": dice.item()
            }
            progress_bar.set_postfix(**batch_metrics)

    # Calculate average metrics
    if total_batches > 0:
        for key in metrics:
            metrics[key] /= total_batches
    return metrics

def convert_image(input_path, output_path, model_path, device):
    """Convert a single input image to segmentation mask using trained model"""
    if not os.path.exists(input_path) or not os.path.exists(model_path):
        print(f"Error: Input file or model file not found.")
        return False

    # Load model
    model = SimpleUNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define transforms for input image
    image_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Load and process image
        original_image = Image.open(input_path).convert("RGB")
        original_size = original_image.size
        image = image_transforms(original_image).unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            output = model(image)
            mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255

        # Save result
        mask_image = Image.fromarray(mask, mode='L')
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        mask_image.save(output_path)
        print(f"Mask saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error in conversion: {e}")
        return False

def train_model(metrics_csv):
    """Main training loop that handles multi-epoch training and model saving"""
    global stop_training

    # Setup device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    print(f"Using device: {device}" +
          (f" ({torch.cuda.device_count()} GPUs)" if multi_gpu else ""))

    # Set seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Prepare data and model
    train_loader, val_loader, test_loader = prepare_data()
    model = SimpleUNet(n_channels=3, n_classes=1)

    # Try to load existing model
    if best_model_path_path.exists():
        try:
            model.load_state_dict(torch.load(best_model_path_path, map_location=device))
            print(f"Loaded existing model from {best_model_path_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    # Setup parallel processing if multiple GPUs available
    if multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    # Define loss, optimizer and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)

    best_val_iou = -1.0
    start_time = time.time()

    # Training loop
    for epoch in range(1, 9999):
        if stop_training:
            break

        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader,
            optimizer, criterion, device, epoch, metrics_csv)
        print(f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.4f}, "
              f"IoU={train_metrics['iou']:.4f}, Dice={train_metrics['dice']:.4f}")

        if stop_training:
            break

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device, phase="val")
        print(f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.4f}, "
              f"IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}")

        # Adjust learning rate based on performance
        scheduler.step(val_metrics['iou'])

        # Save best model based on IoU score
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            model_state = model.module.state_dict() if multi_gpu else model.state_dict()
            torch.save(model_state, best_model_path_path)
            print(f"Saved new best model (IoU: {best_val_iou:.4f})")

    print(f"Training finished in {(time.time() - start_time) / 60:.2f} minutes")
    return test_loader

def test_model(model_path, device, test_loader=None):
    """Evaluate model on test dataset"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load test data if not provided
    if test_loader is None:
        _, _, test_loader = prepare_data()

    # Load model
    model = SimpleUNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate and report results
    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, device, phase="test")

    print("--- Test Results ---")
    print(f" Loss: {test_metrics['loss']:.4f}")
    print(f" Acc : {test_metrics['acc']:.4f}")
    print(f" IoU : {test_metrics['iou']:.4f}")
    print(f" Dice: {test_metrics['dice']:.4f}")

if __name__ == "__main__":
    # Setup paths
    base_data_dir = Path(__file__).parent
    image_dir_path = base_data_dir / image_dir
    mask_dir_path = base_data_dir / mask_dir
    best_model_path_path = base_data_dir / best_model_path

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="U-Net for Track Limit Segmentation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--test", action="store_true", help="Test the model")
    group.add_argument("--convert", nargs=2, metavar=('INPUT_PATH', 'OUTPUT_PATH'),
        help="Convert an input image to mask")
    parser.add_argument("--model", type=str, default=str(best_model_path_path),
        help=f"Model path (default: {best_model_path_path})")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_csv_path = Path(base_data_dir / metrics_csv_path)

    # Execute requested operation
    if args.convert:
        convert_image(args.convert[0], args.convert[1], args.model, device)
    elif args.test:
        test_model(args.model, device)
    elif args.train:
        test_loader = train_model(metrics_csv_path)
        if not stop_training:
            test_model(best_model_path_path, device, test_loader)
