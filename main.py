import torch
import torch.nn as nn
import torch.optim as optim
from mpmath import monitor
from torch.onnx.symbolic_helper import check_training_mode
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
from pathlib import Path
import numpy as np
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import shutil
from pathlib import WindowsPath
import time
from typing import Optional
from rembg import remove, new_session
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# class BackgroundProcessor:
#     def __init__(self, threshold=0.5, blur_kernel_size=5, smooth_factor=0.8):
#         self.threshold = threshold
#         self.blur_kernel_size = blur_kernel_size
#         self.smooth_factor = smooth_factor
#
#     def apply_gaussian_smoothing(self, tensor):
#         padding = (self.blur_kernel_size - 1) // 2
#         gaussian_kernel = self._create_gaussian_kernel(self.blur_kernel_size)
#         gaussian_kernel = gaussian_kernel.expand(tensor.size(1), 1, self.blur_kernel_size, self.blur_kernel_size)
#
#         return F.conv2d(
#             tensor,
#             gaussian_kernel.to(tensor.device),
#             padding=padding,
#             groups=tensor.size(1)
#         )
#
#     def _create_gaussian_kernel(self, kernel_size, sigma=1.5):
#         x = torch.linspace(-sigma, sigma, kernel_size)
#         x = x.expand(kernel_size, -1)
#         y = x.t()
#
#         kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#         kernel = kernel / kernel.sum()
#
#         return kernel.unsqueeze(0).unsqueeze(0)
#
#     def get_background_mask(self, tensor):
#         # Ensure input tensor has correct dimensions
#         B, C, H, W = tensor.size()
#
#         # Convert to grayscale
#         gray = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
#
#         # Calculate gradients
#         dx = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 1, 0, 0), mode='replicate')
#         dy = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 0, 1), mode='replicate')
#
#         # Calculate gradient magnitude
#         gradient_mag = torch.sqrt(dx ** 2 + dy ** 2)
#
#         # Create mask
#         mask = (gradient_mag < self.threshold).float()
#
#         # Apply smoothing
#         mask = self.apply_gaussian_smoothing(mask.unsqueeze(1))
#
#         # Ensure mask has same size as input
#         mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
#
#         return mask.expand(-1, 3, -1, -1)
#
#     def process_image(self, tensor, background_type='smooth'):
#         B, C, H, W = tensor.size()
#         bg_mask = self.get_background_mask(tensor)
#
#         if background_type == 'smooth':
#             bg = torch.ones_like(tensor) * 0.5
#         elif background_type == 'gradient':
#             gradient = torch.linspace(0, 1, W, device=tensor.device)
#             gradient = gradient.view(1, 1, 1, -1).expand(B, C, H, W)
#             bg = gradient
#         else:  # pattern
#             bg = torch.zeros_like(tensor)
#             bg[:, :, ::4, ::4] = 1
#
#         # Ensure all tensors have the same size
#         processed = (1 - bg_mask) * tensor + bg_mask * bg
#         processed = self.apply_gaussian_smoothing(processed)
#
#         return processed
#
#     def create_smooth_background(self, tensor):
#         return torch.ones_like(tensor) * 0.5
#
#     def create_gradient_background(self, tensor):
#         h, w = tensor.shape[2:]
#         gradient = torch.linspace(0, 1, w, device=tensor.device)
#         gradient = gradient.view(1, 1, 1, -1).expand(tensor.shape[0], 3, h, w)
#         return gradient
#
#     def create_pattern_background(self, tensor):
#         h, w = tensor.shape[2:]
#         pattern = torch.zeros_like(tensor)
#         pattern[:, :, ::4, ::4] = 1
#         return pattern

class BackgroundProcessor:
    def __init__(self):
        pass

    def process_image(self, tensor, background_type=None):
        # Simply return the tensor unchanged
        return tensor

    def create_smooth_background(self, tensor):
        # Return unchanged
        return tensor

    def create_pattern_background(self, tensor):
        # Return unchanged
        return tensor

class NetworkPathHandler:
    def __init__(self, network_path: str, max_retries: int = 3, retry_delay: int = 5):
        self.network_path = Path(network_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def check_path_accessible(self) -> bool:
        try:
            return self.network_path.exists()
        except (PermissionError, OSError):
            return False

    def wait_for_access(self) -> bool:
        for attempt in range(self.max_retries):
            if self.check_path_accessible():
                return True
            logger.warning(f"Network path not accessible, attempt {attempt + 1}/{self.max_retries}")
            time.sleep(self.retry_delay)
        return False


class AnimeDataset(Dataset):
    def __init__(self, network_path: str, image_size: int = 256, remove_bg: bool = True, store_originals: bool = True):
        self.network_handler = NetworkPathHandler(network_path)
        if not self.network_handler.wait_for_access():
            raise RuntimeError(f"Could not access network path: {network_path}")

        self.root_dir = Path(network_path)
        self.image_size = image_size
        self.remove_bg = remove_bg
        self.store_originals = store_originals

        # Initialize rembg session with u2net model
        self.rembg_session = new_session("u2net")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Initialize cache directories
        self.cache_dir = Path("processed_cache")
        self.nobg_cache_dir = Path("nobg_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.nobg_cache_dir.mkdir(exist_ok=True)

        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        logger.info("Scanning network directory for images...")
        self._scan_directories()
        self._process_and_cache_images()

    def _scan_directories(self):
        """Scan directories with error handling for network issues"""
        try:
            for franchise_dir in self.root_dir.iterdir():
                if franchise_dir.is_dir():
                    for char_dir in franchise_dir.iterdir():
                        if char_dir.is_dir():
                            label = f"{franchise_dir.name}/{char_dir.name}"
                            if label not in self.label_to_idx:
                                idx = len(self.label_to_idx)
                                self.label_to_idx[label] = idx
                                self.idx_to_label[idx] = label

                            for img_path in char_dir.glob("*.[jp][pn][g]"):
                                self.image_paths.append(img_path)
                                self.labels.append(self.label_to_idx[label])

            if not self.image_paths:
                raise RuntimeError("No images found in the specified directory")

            logger.info(f"Found {len(self.image_paths)} images across {len(self.label_to_idx)} categories")

        except Exception as e:
            logger.error(f"Error scanning network directory: {e}")
            raise RuntimeError("Failed to scan network directory")

    def _remove_background(self, img_path: Path) -> tuple[Image.Image, Image.Image]:
        """Remove background from image using rembg and return both versions"""
        try:
            # Generate cache paths for both versions
            orig_cache_path = self.cache_dir / f"{img_path.stem}_orig.png"
            nobg_cache_path = self.nobg_cache_dir / f"{img_path.stem}_nobg.png"

            # Check if both cached versions exist
            if orig_cache_path.exists() and nobg_cache_path.exists():
                return Image.open(orig_cache_path), Image.open(nobg_cache_path)

            # Process image
            with Image.open(img_path) as img:
                # Convert to RGB and save original
                orig_img = img.convert('RGB')
                orig_img.save(orig_cache_path)

                # Remove background with rembg
                if self.remove_bg:
                    # Convert to RGBA for transparency
                    img_rgba = img.convert('RGBA')

                    # Use rembg to remove background
                    nobg_img = remove(
                        img_rgba,
                        session=self.rembg_session,
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10
                    )

                    # Create white background
                    background = Image.new('RGBA', nobg_img.size, (255, 255, 255, 255))
                    background.paste(nobg_img, mask=nobg_img.split()[-1])
                    nobg_img = background.convert('RGB')

                    # Center crop the character
                    nobg_img = self._center_crop_character(nobg_img)
                else:
                    nobg_img = orig_img.copy()

                nobg_img.save(nobg_cache_path)
                return orig_img, nobg_img

        except Exception as e:
            logger.warning(f"Failed to remove background from {img_path}: {e}")
            # Return original image for both if processing fails
            with Image.open(img_path) as img:
                orig_img = img.convert('RGB')
                return orig_img, orig_img

    def _process_and_cache_images(self):
        """Pre-process and cache all images with progress bar"""
        logger.info("Processing and caching images with rembg background removal...")
        total_images = len(self.image_paths)

        # Process images in batches to optimize memory usage
        batch_size = 10
        for i in range(0, total_images, batch_size):
            batch_paths = self.image_paths[i:i + batch_size]
            with tqdm(total=len(batch_paths),
                      desc=f"Processing batch {i // batch_size + 1}/{(total_images + batch_size - 1) // batch_size}") as pbar:
                for img_path in batch_paths:
                    try:
                        self._remove_background(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to process {img_path}: {e}")
                    pbar.update(1)

            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _center_crop_character(self, img: Image.Image) -> Image.Image:
        """Center crop the image around the non-transparent areas"""
        # If image is RGBA, use alpha channel for cropping
        if img.mode == 'RGBA':
            alpha = np.array(img.split()[-1])
            non_transparent = np.where(alpha > 0)

            if len(non_transparent[0]) > 0:
                # Get bounding box
                top = max(0, np.min(non_transparent[0]))
                left = max(0, np.min(non_transparent[1]))
                bottom = min(img.height, np.max(non_transparent[0]))
                right = min(img.width, np.max(non_transparent[1]))

                # Add padding to maintain aspect ratio
                height = bottom - top
                width = right - left
                max_dim = max(height, width)

                # Calculate padding to make square
                pad_vert = (max_dim - height) // 2
                pad_horz = (max_dim - width) // 2

                # Adjust bounds with padding
                top = max(0, top - pad_vert)
                bottom = min(img.height, bottom + pad_vert)
                left = max(0, left - pad_horz)
                right = min(img.width, right + pad_horz)

                # Crop image
                cropped = img.crop((left, top, right, bottom))

                # Resize to maintain consistent size
                return cropped.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        return img

    def _validate_and_process_images(self):
        """Validate and process all images"""
        valid_paths = []
        valid_labels = []

        logger.info("Validating and processing images...")
        for idx, (path, label) in enumerate(zip(self.image_paths, self.labels)):
            try:
                if self._is_valid_image(path):
                    if self.remove_bg:
                        # Process images with rembg
                        orig_img, nobg_img = self._remove_background(path)

                        # Validate processed images
                        if nobg_img is not None:
                            valid_paths.append(path)
                            valid_labels.append(label)
                    else:
                        valid_paths.append(path)
                        valid_labels.append(label)

                if idx % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(self.image_paths)} images")

            except Exception as e:
                logger.warning(f"Failed to process image {path}: {e}")
                continue

        self.image_paths = valid_paths
        self.labels = valid_labels
        logger.info(f"Kept {len(valid_paths)} valid images out of {len(self.image_paths)} total")

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img_path = self.image_paths[idx]

                # Load both versions from cache
                orig_cache_path = self.cache_dir / f"{Path(img_path).stem}_orig.png"
                nobg_cache_path = self.nobg_cache_dir / f"{Path(img_path).stem}_nobg.png"

                if self.store_originals:
                    # Return both original and no-background versions
                    orig_img = Image.open(orig_cache_path)
                    nobg_img = Image.open(nobg_cache_dir)

                    # Apply transformations
                    orig_tensor = self.transform(orig_img)
                    nobg_tensor = self.transform(nobg_img)

                    return orig_tensor, nobg_tensor, self.labels[idx]
                else:
                    # Return only no-background version
                    nobg_img = Image.open(nobg_cache_path)
                    nobg_tensor = self.transform(nobg_img)
                    return nobg_tensor, self.labels[idx]

            except (OSError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to load image {img_path}: {str(e)}")
                if attempt == max_retries - 1:
                    return self.__getitem__((idx + 1) % len(self))
                time.sleep(1)

    def __len__(self):
        return len(self.image_paths)


import logging
from pathlib import Path
import time
from datetime import datetime
import json
import psutil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.utils as vutils
from typing import List, Dict, Optional, Union
import pandas as pd
import seaborn as sns


class ResourceMonitor:
    """Monitors system and GPU resources during training."""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'timestamps': []
        }

    def update(self) -> Dict[str, float]:
        """Collect current resource usage metrics."""
        current_metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }

        if self.gpu_available:
            gpu_stats = torch.cuda.get_device_properties(0)
            current_metrics.update({
                'gpu_usage': torch.cuda.utilization(),
                'gpu_memory': torch.cuda.memory_allocated() / gpu_stats.total_memory * 100
            })

        # Update historical metrics
        for key, value in current_metrics.items():
            if key != 'timestamp':
                self.metrics[key].append(value)
        self.metrics['timestamps'].append(current_metrics['timestamp'])

        return current_metrics


class TrainingLogger:
    """Handles logging of training metrics and events."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Set up file logging
        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Set up TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))

        # Initialize CSV logger
        self.csv_path = self.log_dir / 'metrics.csv'
        self.metrics_df = pd.DataFrame()

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all outputs (file, TensorBoard, CSV)."""
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

        # Log to CSV
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_df = pd.concat([
            self.metrics_df,
            pd.DataFrame([metrics])
        ], ignore_index=True)
        self.metrics_df.to_csv(self.csv_path, index=False)

        # Log to file
        logging.info(f"Step {step}: {json.dumps(metrics)}")


class EnhancedTrainingMonitor:
    """Enhanced training monitor with comprehensive logging and visualization."""

    def __init__(self, save_dir: str = "training_progress"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize components
        self.logger = TrainingLogger(self.save_dir / 'logs')
        self.resource_monitor = ResourceMonitor()

        # Training metrics
        self.metrics = {
            'g_losses': [],
            'd_losses': [],
            'fid_scores': [],
            'inception_scores': []
        }

        self.step = 0

    def update(self, g_loss: float, d_loss: float,
               fid_score: Optional[float] = None,
               inception_score: Optional[float] = None):
        """Update training metrics and resource usage."""
        # Update training metrics
        metrics = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        if fid_score is not None:
            metrics['fid_score'] = fid_score
        if inception_score is not None:
            metrics['inception_score'] = inception_score

        # Update resource metrics
        resource_metrics = self.resource_monitor.update()
        metrics.update(resource_metrics)

        # Log everything
        self.logger.log_metrics(metrics, self.step)

        # Store for internal tracking
        self.metrics['g_losses'].append(g_loss)
        self.metrics['d_losses'].append(d_loss)
        if fid_score is not None:
            self.metrics['fid_scores'].append(fid_score)
        if inception_score is not None:
            self.metrics['inception_scores'].append(inception_score)

        self.step += 1

    def plot_training_progress(self, save: bool = True) -> None:
        """Generate comprehensive training progress plots."""
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        grid = plt.GridSpec(2, 2, figure=fig)

        # Plot losses
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.plot(self.metrics['g_losses'], label='Generator Loss')
        ax1.plot(self.metrics['d_losses'], label='Discriminator Loss')
        ax1.set_title('Training Losses')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot resource usage
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.plot(self.resource_monitor.metrics['cpu_usage'], label='CPU Usage (%)')
        if self.resource_monitor.gpu_available:
            ax2.plot(self.resource_monitor.metrics['gpu_usage'], label='GPU Usage (%)')
        ax2.set_title('Resource Usage')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Usage (%)')
        ax2.legend()

        # Plot memory usage
        ax3 = fig.add_subplot(grid[1, 0])
        ax3.plot(self.resource_monitor.metrics['memory_usage'], label='System Memory')
        if self.resource_monitor.gpu_available:
            ax3.plot(self.resource_monitor.metrics['gpu_memory'], label='GPU Memory')
        ax3.set_title('Memory Usage')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Usage (%)')
        ax3.legend()

        # Plot quality metrics if available
        ax4 = fig.add_subplot(grid[1, 1])
        if self.metrics['fid_scores']:
            ax4.plot(self.metrics['fid_scores'], label='FID Score')
        if self.metrics['inception_scores']:
            ax4.plot(self.metrics['inception_scores'], label='Inception Score')
        ax4.set_title('Quality Metrics')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Score')
        ax4.legend()

        plt.tight_layout()

        if save:
            plt.savefig(self.save_dir / f'training_progress_{self.step}.png')
            plt.close()
        else:
            plt.show()

    def save_samples(self, images: torch.Tensor, prefix: str = 'generated'):
        """Save generated image samples with metadata."""
        # Save image grid
        grid = vutils.make_grid(images, normalize=True)
        vutils.save_image(grid, self.save_dir / f'{prefix}_samples_step_{self.step}.png')

        # Log sample to TensorBoard
        self.logger.writer.add_image(f'{prefix}_samples', grid, self.step)

    def check_training_quality(self) -> List[str]:
        """Enhanced training quality checks."""
        issues = []

        if len(self.metrics['g_losses']) > 100:
            recent_g_loss = np.mean(self.metrics['g_losses'][-20:])
            initial_g_loss = np.mean(self.metrics['g_losses'][:20])

            # Check various training issues
            if recent_g_loss > initial_g_loss * 1.5:
                issues.append("Generator loss is significantly increasing")

            if np.std(self.metrics['g_losses'][-20:]) < 0.01:
                issues.append("Possible mode collapse detected")

            recent_d_loss = np.mean(self.metrics['d_losses'][-20:])
            if recent_d_loss < 0.1:
                issues.append("Discriminator may be too strong")

            if abs(recent_g_loss - self.metrics['g_losses'][-2]) < 1e-7:
                issues.append("Possible vanishing gradients")

            # Resource-related issues
            if self.resource_monitor.gpu_available:
                recent_gpu_usage = np.mean(self.resource_monitor.metrics['gpu_usage'][-20:])
                if recent_gpu_usage > 95:
                    issues.append("GPU usage consistently very high")

            recent_memory_usage = np.mean(self.resource_monitor.metrics['memory_usage'][-20:])
            if recent_memory_usage > 90:
                issues.append("System memory usage critically high")

        return issues

    def save_checkpoint(self):
        """Save all monitoring data to disk."""
        checkpoint = {
            'metrics': self.metrics,
            'resource_metrics': self.resource_monitor.metrics,
            'step': self.step
        }
        torch.save(checkpoint, self.save_dir / f'monitor_checkpoint_{self.step}.pt')

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load monitoring data from a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.metrics = checkpoint['metrics']
        self.resource_monitor.metrics = checkpoint['resource_metrics']
        self.step = checkpoint['step']

# Improved Generator Network with Self-Attention and Better Architecture
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 100)

        self.initial = nn.Sequential(
            nn.Linear(latent_dim + 100, 512 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.Dropout(0.3)
        )

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.3)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.3)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2)
            )
        ])

        self.attention = SelfAttention(128)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, z, labels):
        # Improved label conditioning
        label_embedding = self.label_embedding(labels)
        x = torch.cat([z, label_embedding], dim=1)

        # Initial dense layer
        x = self.initial(x)
        x = x.view(-1, 512, 8, 8)

        # Process through conv blocks with attention
        for idx, block in enumerate(self.conv_blocks):
            x = block(x)
            if idx == 1:  # Apply attention after second block
                x = self.attention(x)

        # Final convolution
        x = self.final(x)
        return x


# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Reshape for attention
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)

        # Calculate attention
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=2)

        # Apply attention to value
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return x + self.gamma * out


# Modified training parameters
def get_improved_training_params():
    return {
        'g_lr': 0.0001,  # Reduced learning rate for stability
        'd_lr': 0.0004,  # Higher discriminator learning rate
        'beta1': 0.5,
        'beta2': 0.999,
        'latent_dim': 128,  # Increased from 100
        'label_smoothing': 0.1,  # Add label smoothing
        'noise_factor': 0.05  # For noisy labels
    }


# Training improvements to add to AnimeGeneratorTrainer.__init__
def improved_training_setup(self):
    params = get_improved_training_params()

    # Use different learning rates for G and D
    self.g_optimizer = optim.Adam(
        self.generator.parameters(),
        lr=params['g_lr'],
        betas=(params['beta1'], params['beta2'])
    )
    self.d_optimizer = optim.Adam(
        self.discriminator.parameters(),
        lr=params['d_lr'],
        betas=(params['beta1'], params['beta2'])
    )

    # Add learning rate schedulers
    self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.g_optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.d_optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Modified label values with smoothing
    self.real_label_val = 0.9  # Instead of 1.0
    self.fake_label_val = 0.1  # Instead of 0.0


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 50)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        self.output = nn.Sequential(
            nn.Linear(256 * 8 * 8 + 50, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        features = self.model(x)
        label_embedding = self.label_embedding(labels)
        combined = torch.cat([features, label_embedding], dim=1)
        return self.output(combined)


class AnimeGeneratorTrainer:
    def __init__(self, network_path: str, batch_size: int = 64, latent_dim: int = 100,
                 lr: float = 0.0002, image_size: int = 128, local_cache_dir: Optional[str] = None,
                 store_originals: bool = True):
        self.background_processor = BackgroundProcessor()

        # Check CUDA availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda:0")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA is not available. Running on CPU!")

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self.store_originals = store_originals

        # Initialize dataset with background removal
        self.dataset = AnimeDataset(
            network_path,
            image_size,
            remove_bg=True,
            store_originals=store_originals
        )

        num_workers = 4 if self.use_cuda else 0
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Initialize networks
        self.generator = Generator(latent_dim, len(self.dataset.label_to_idx))
        self.discriminator = Discriminator(len(self.dataset.label_to_idx))

        # Move models to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr * 1.5, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.criterion = nn.BCEWithLogitsLoss()
        self.real_label_val = 0.9
        self.fake_label_val = 0.0
        self.monitor = EnhancedTrainingMonitor()

        # Log GPU memory usage after initialization
        if torch.cuda.is_available():
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    def monitor_training_health(self, g_loss, d_loss, epoch, batch_idx):
        """Monitor training health and provide warnings"""
        warnings = []

        # Check discriminator strength
        if d_loss < 0.1:
            warnings.append("Warning: Discriminator may be too strong")
        elif d_loss > 2.0:
            warnings.append("Warning: Discriminator may be too weak")

        # Check generator performance
        if g_loss > 5.0:
            warnings.append("Warning: Generator loss too high")
        elif g_loss < 0.1:
            warnings.append("Warning: Possible mode collapse")

        # Check loss ratio
        loss_ratio = g_loss / (d_loss + 1e-8)
        if loss_ratio > 30:
            warnings.append("Warning: Generator/Discriminator loss ratio too high")
        elif loss_ratio < 0.1:
            warnings.append("Warning: Generator/Discriminator loss ratio too low")

        if warnings:
            logger.warning(f"Epoch {epoch}, Batch {batch_idx}: {'; '.join(warnings)}")

        return len(warnings) == 0  # Return True if training looks healthy

    def generate_samples(self, num_samples, labels, background_type='smooth'):
        """Generate samples with background processing"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake_images = self.generator(noise, labels)
            processed_images = self.background_processor.process_image(
                fake_images,
                background_type=background_type
            )
        self.generator.train()
        return processed_images

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            return checkpoint['epoch']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def save_checkpoint(self, epoch: int, path: str = "checkpoints", is_best: bool = False):
        try:
            os.makedirs(path, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'label_to_idx': self.dataset.label_to_idx,
                'g_losses': self.monitor.g_losses,
                'd_losses': self.monitor.d_losses
            }

            filename = f"checkpoint_epoch_{epoch}.pt"
            if is_best:
                filename = "best_model.pt"

            save_path = os.path.join(path, filename)
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def train_step(self, data, labels):
        """Modified training step to handle both image versions"""
        batch_size = labels.size(0)

        # Unpack data based on whether we're storing originals
        if self.store_originals:
            real_orig, real_nobg, labels = data
            # Train on no-background images
            real_images = real_nobg
        else:
            real_images, labels = data

        # Train discriminator
        self.d_optimizer.zero_grad()

        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise, labels)

        d_loss_real = self.criterion(
            self.discriminator(real_images, labels),
            torch.ones(batch_size, 1, device=self.device) * self.real_label_val
        )
        d_loss_fake = self.criterion(
            self.discriminator(fake_images.detach(), labels),
            torch.zeros(batch_size, 1, device=self.device)
        )

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.g_optimizer.zero_grad()
        g_loss = self.criterion(
            self.discriminator(fake_images, labels),
            torch.ones(batch_size, 1, device=self.device) * self.real_label_val
        )
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

    def train(self, num_epochs=500, save_interval=5):
        for epoch in range(num_epochs):
            running_d_loss = 0.0
            running_g_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, batch_data in enumerate(pbar):
                if self.store_originals:
                    real_orig, real_nobg, labels = [item.to(self.device) for item in batch_data]
                    d_loss, g_loss = self.train_step((real_orig, real_nobg, labels), labels)
                else:
                    real_images, labels = [item.to(self.device) for item in batch_data]
                    d_loss, g_loss = self.train_step((real_images, labels), labels)

                running_d_loss += d_loss
                running_g_loss += g_loss

                if batch_idx % 10 == 0:
                    self.monitor.update(g_loss, d_loss)
                    training_healthy = self.monitor_training_health(g_loss, d_loss, epoch + 1, batch_idx)

                    status = {
                        'D_loss': f"{d_loss:.4f}",
                        'G_loss': f"{g_loss:.4f}",
                        'Healthy': training_healthy
                    }

                    if self.use_cuda:
                        torch.cuda.empty_cache()
                        status['GPU_mem'] = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"

                    pbar.set_postfix(status)

                if batch_idx % 100 == 0 or batch_idx == len(self.dataloader) - 1:
                    self.save_samples(epoch, batch_idx)

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
                self.monitor.plot_training_progress(save=True)

            avg_d_loss = running_d_loss / len(self.dataloader)
            avg_g_loss = running_g_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1} - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")

def main():
    # Configuration
    network_path = r"\\192.168.1.66\plex\hentai\processed_images"
    local_cache_dir = "local_cache"
    batch_size = 32
    num_epochs = 500
    save_interval = 5
    image_size = 128
    latent_dim = 100

    trainer = AnimeGeneratorTrainer(
        network_path=network_path,
        batch_size=batch_size,
        latent_dim=latent_dim,
        image_size=image_size,
        local_cache_dir=local_cache_dir
    )

    trainer.train(num_epochs=num_epochs, save_interval=save_interval)

    # Generate samples with different backgrounds
    logger.info("Generating sample images with different backgrounds...")
    trainer.load_checkpoint("checkpoints/best_model.pt")

    num_samples = min(16, len(trainer.dataset.label_to_idx))
    sample_labels = torch.arange(num_samples).to(trainer.device)

    # Generate with different background types
    for bg_type in ['smooth', 'gradient', 'pattern']:
        generated_images = trainer.generate_samples(
            num_samples,
            sample_labels,
            background_type=bg_type
        )
        trainer.monitor.save_samples(
            generated_images,
            num_epochs,
            prefix=f'final_{bg_type}_bg'
        )

    logger.info(f"Final samples saved to {trainer.monitor.save_dir}")

if __name__ == "__main__":
    main()