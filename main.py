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
from rembg import remove
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundProcessor:
    def __init__(self, threshold=0.5, blur_kernel_size=5, smooth_factor=0.8):
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.smooth_factor = smooth_factor

    def apply_gaussian_smoothing(self, tensor):
        padding = (self.blur_kernel_size - 1) // 2
        gaussian_kernel = self._create_gaussian_kernel(self.blur_kernel_size)
        gaussian_kernel = gaussian_kernel.expand(tensor.size(1), 1, self.blur_kernel_size, self.blur_kernel_size)

        return F.conv2d(
            tensor,
            gaussian_kernel.to(tensor.device),
            padding=padding,
            groups=tensor.size(1)
        )

    def _create_gaussian_kernel(self, kernel_size, sigma=1.5):
        x = torch.linspace(-sigma, sigma, kernel_size)
        x = x.expand(kernel_size, -1)
        y = x.t()

        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        return kernel.unsqueeze(0).unsqueeze(0)

    def get_background_mask(self, tensor):
        # Ensure input tensor has correct dimensions
        B, C, H, W = tensor.size()

        # Convert to grayscale
        gray = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]

        # Calculate gradients
        dx = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 1, 0, 0), mode='replicate')
        dy = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 0, 1), mode='replicate')

        # Calculate gradient magnitude
        gradient_mag = torch.sqrt(dx ** 2 + dy ** 2)

        # Create mask
        mask = (gradient_mag < self.threshold).float()

        # Apply smoothing
        mask = self.apply_gaussian_smoothing(mask.unsqueeze(1))

        # Ensure mask has same size as input
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        return mask.expand(-1, 3, -1, -1)

    def process_image(self, tensor, background_type='smooth'):
        B, C, H, W = tensor.size()
        bg_mask = self.get_background_mask(tensor)

        if background_type == 'smooth':
            bg = torch.ones_like(tensor) * 0.5
        elif background_type == 'gradient':
            gradient = torch.linspace(0, 1, W, device=tensor.device)
            gradient = gradient.view(1, 1, 1, -1).expand(B, C, H, W)
            bg = gradient
        else:  # pattern
            bg = torch.zeros_like(tensor)
            bg[:, :, ::4, ::4] = 1

        # Ensure all tensors have the same size
        processed = (1 - bg_mask) * tensor + bg_mask * bg
        processed = self.apply_gaussian_smoothing(processed)

        return processed

    def create_smooth_background(self, tensor):
        return torch.ones_like(tensor) * 0.5

    def create_gradient_background(self, tensor):
        h, w = tensor.shape[2:]
        gradient = torch.linspace(0, 1, w, device=tensor.device)
        gradient = gradient.view(1, 1, 1, -1).expand(tensor.shape[0], 3, h, w)
        return gradient

    def create_pattern_background(self, tensor):
        h, w = tensor.shape[2:]
        pattern = torch.zeros_like(tensor)
        pattern[:, :, ::4, ::4] = 1
        return pattern

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
    def __init__(self, network_path: str, image_size: int = 256, remove_bg: bool = True):
        self.network_handler = NetworkPathHandler(network_path)
        if not self.network_handler.wait_for_access():
            raise RuntimeError(f"Could not access network path: {network_path}")

        self.root_dir = Path(network_path)
        self.image_size = image_size
        self.remove_bg = remove_bg

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        self.cache_dir = Path("processed_cache")
        self.cache_dir.mkdir(exist_ok=True)

        logger.info("Scanning network directory for images...")
        self._scan_directories()
        # self._validate_and_process_images()  # Commented out validation

    """
    def _remove_background(self, img_path: Path) -> Image.Image:
        try:
            cache_path = self.cache_dir / f"{img_path.stem}_nobg.png"

            if cache_path.exists():
                return Image.open(cache_path)

            with Image.open(img_path) as img:
                img = img.convert('RGBA')
                output = remove(img)
                output.save(cache_path)
                return output

        except Exception as e:
            logger.warning(f"Failed to remove background from {img_path}: {e}")
            return Image.open(img_path)

    def _center_crop_character(self, img: Image.Image) -> Image.Image:
        if img.mode == 'RGBA':
            alpha = np.array(img.split()[-1])
            non_transparent = np.where(alpha > 0)
            if len(non_transparent[0]) > 0:
                top, left = np.min(non_transparent[0]), np.min(non_transparent[1])
                bottom, right = np.max(non_transparent[0]), np.max(non_transparent[1])

                height, width = bottom - top, right - left
                padding = max(height, width) // 4

                top = max(0, top - padding)
                bottom = min(img.height, bottom + padding)
                left = max(0, left - padding)
                right = min(img.width, right + padding)

                return img.crop((left, top, right, bottom))
        return img

    def _validate_and_process_images(self):
        valid_paths = []
        valid_labels = []

        logger.info("Validating and processing images...")
        for idx, (path, label) in enumerate(zip(self.image_paths, self.labels)):
            try:
                if self._is_valid_image(path):
                    if self.remove_bg:
                        processed_img = self._remove_background(path)
                        processed_img = self._center_crop_character(processed_img)

                        if processed_img.mode == 'RGBA':
                            background = Image.new('RGBA', processed_img.size, (255, 255, 255, 255))
                            background.paste(processed_img, mask=processed_img.split()[-1])
                            processed_img = background.convert('RGB')

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

    def _validate_images(self):
        valid_paths = []
        valid_labels = []

        logger.info("Validating images...")
        for idx, (path, label) in enumerate(zip(self.image_paths, self.labels)):
            if self._is_valid_image(path):
                valid_paths.append(path)
                valid_labels.append(label)

            if idx % 100 == 0:
                logger.info(f"Validated {idx + 1}/{len(self.image_paths)} images")

        self.image_paths = valid_paths
        self.labels = valid_labels
        logger.info(f"Kept {len(valid_paths)} valid images out of {len(self.image_paths)} total")

    def _is_valid_image(self, path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
                img = Image.open(path).convert('RGB')
                return True
        except Exception as e:
            logger.warning(f"Invalid image file {path}: {str(e)}")
            return False
    """

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

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img_path = self.image_paths[idx]

                if self.remove_bg:
                    # Load processed image from cache
                    cache_path = self.cache_dir / f"{Path(img_path).stem}_nobg.png"
                    if cache_path.exists():
                        img = Image.open(cache_path)
                    else:
                        # img = self._remove_background(img_path)  # Commented out background removal
                        # img = self._center_crop_character(img)   # Commented out center cropping
                        img = Image.open(img_path)
                else:
                    img = Image.open(img_path)

                # Convert to RGB
                if img.mode == 'RGBA':
                    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background.convert('RGB')
                else:
                    img = img.convert('RGB')

                # Apply transformations
                image = self.transform(img)
                label = self.labels[idx]
                return image, label

            except (OSError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to load image {img_path}: {str(e)}")
                if attempt == max_retries - 1:
                    new_idx = (idx + 1) % len(self)
                    return self.__getitem__(new_idx)
                time.sleep(1)

    def __len__(self):
        return len(self.image_paths)


class TrainingMonitor:
    def __init__(self, save_dir="training_progress", log_interval=10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "metrics").mkdir(exist_ok=True)

        # Metrics tracking
        self.g_losses = []
        self.d_losses = []
        self.g_running_avg = []
        self.d_running_avg = []
        self.learning_rates = {'g': [], 'd': []}
        self.quality_metrics = {'healthy_ratio': [], 'mode_collapse_score': []}

        # Training stability metrics
        self.gradient_norms = {'g': [], 'd': []}
        self.loss_ratios = []
        self.log_interval = log_interval

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        # Configure detailed logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        # File handler
        fh = logging.FileHandler(self.save_dir / 'training.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)

        # Stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)

        # Get logger
        self.logger = logging.getLogger('GANTraining')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def update(self, g_loss, d_loss, g_lr, d_lr, g_norm=None, d_norm=None):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        self.learning_rates['g'].append(g_lr)
        self.learning_rates['d'].append(d_lr)

        if g_norm is not None and d_norm is not None:
            self.gradient_norms['g'].append(g_norm)
            self.gradient_norms['d'].append(d_norm)

        # Calculate running averages
        window = min(50, len(self.g_losses))
        if window > 0:
            self.g_running_avg.append(np.mean(self.g_losses[-window:]))
            self.d_running_avg.append(np.mean(self.d_losses[-window:]))

        # Calculate loss ratio
        if d_loss != 0:
            self.loss_ratios.append(g_loss / d_loss)

    def plot_training_progress(self, epoch):
        plt.figure(figsize=(20, 12))

        # Plot 1: Losses
        plt.subplot(2, 2, 1)
        plt.plot(self.g_losses[-1000:], label='Generator Loss', alpha=0.3)
        plt.plot(self.d_losses[-1000:], label='Discriminator Loss', alpha=0.3)
        plt.plot(self.g_running_avg[-1000:], label='G Running Avg', linewidth=2)
        plt.plot(self.d_running_avg[-1000:], label='D Running Avg', linewidth=2)
        plt.title('Training Losses')
        plt.legend()

        # Plot 2: Learning Rates
        plt.subplot(2, 2, 2)
        plt.plot(self.learning_rates['g'], label='Generator LR')
        plt.plot(self.learning_rates['d'], label='Discriminator LR')
        plt.title('Learning Rates')
        plt.legend()

        # Plot 3: Gradient Norms
        plt.subplot(2, 2, 3)
        plt.plot(self.gradient_norms['g'][-1000:], label='Generator Gradients')
        plt.plot(self.gradient_norms['d'][-1000:], label='Discriminator Gradients')
        plt.title('Gradient Norms')
        plt.legend()

        # Plot 4: Loss Ratios
        plt.subplot(2, 2, 4)
        plt.plot(self.loss_ratios[-1000:], label='G/D Loss Ratio')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        plt.title('Generator/Discriminator Loss Ratio')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / "plots" / f'training_metrics_epoch_{epoch}.png')
        plt.close()

    def check_training_quality(self, g_loss, d_loss):
        """Enhanced training quality checks"""
        issues = []

        # Check recent losses
        recent_g_losses = self.g_losses[-100:] if len(self.g_losses) >= 100 else self.g_losses
        recent_d_losses = self.d_losses[-100:] if len(self.d_losses) >= 100 else self.d_losses

        # Mode collapse detection
        g_loss_std = np.std(recent_g_losses)
        if g_loss_std < 0.01:
            issues.append("WARNING: Possible mode collapse (low G loss variance)")

        # Discriminator dominance
        if np.mean(recent_d_losses) < 0.1:
            issues.append("WARNING: Discriminator too strong")

        # Generator weakness
        if np.mean(recent_g_losses) > 5.0:
            issues.append("WARNING: Generator may be struggling")

        # Loss ratio instability
        if len(self.loss_ratios) > 0:
            recent_ratios = self.loss_ratios[-100:]
            if np.std(recent_ratios) > 2.0:
                issues.append("WARNING: Unstable G/D loss ratio")

        # Gradient vanishing
        if len(self.gradient_norms['g']) > 0:
            recent_g_grads = self.gradient_norms['g'][-100:]
            if np.mean(recent_g_grads) < 0.01:
                issues.append("WARNING: Possible vanishing gradients in Generator")

        return issues

    def log_epoch_summary(self, epoch, avg_g_loss, avg_d_loss, g_lr, d_lr):
        """Log comprehensive epoch summary"""
        summary = [
            f"\n{'=' * 50}",
            f"Epoch {epoch} Summary:",
            f"{'=' * 50}",
            f"Average Losses:",
            f"  Generator: {avg_g_loss:.4f}",
            f"  Discriminator: {avg_d_loss:.4f}",
            f"Learning Rates:",
            f"  Generator: {g_lr:.6f}",
            f"  Discriminator: {d_lr:.6f}",
        ]

        if len(self.gradient_norms['g']) > 0:
            summary.extend([
                f"Gradient Norms (avg last 100):",
                f"  Generator: {np.mean(self.gradient_norms['g'][-100:]):.4f}",
                f"  Discriminator: {np.mean(self.gradient_norms['d'][-100:]):.4f}"
            ])

        if len(self.loss_ratios) > 0:
            summary.extend([
                f"Loss Ratio (G/D):",
                f"  Current: {self.loss_ratios[-1]:.4f}",
                f"  Avg (last 100): {np.mean(self.loss_ratios[-100:]):.4f}"
            ])

        # Add training quality issues
        issues = self.check_training_quality(avg_g_loss, avg_d_loss)
        if issues:
            summary.extend([
                f"Training Issues Detected:",
                *[f"  - {issue}" for issue in issues]
            ])

        summary.append(f"{'=' * 50}")

        # Log the summary
        self.logger.info('\n'.join(summary))


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
                 lr: float = 0.0002, image_size: int = 128, local_cache_dir: Optional[str] = None):
        # First check CUDA
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
        self.background_processor = BackgroundProcessor()

        # Initialize dataset and dataloader first
        self.dataset = AnimeDataset(network_path, image_size)
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

        # Initialize optimizers AFTER models are created and moved to device
        params = get_improved_training_params()
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

        self.criterion = nn.BCELoss()
        self.real_label_val = 0.9  # Using label smoothing
        self.fake_label_val = 0.0
        self.monitor = TrainingMonitor()

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

    class TrainingMonitor:
        def __init__(self, save_dir="training_progress", log_interval=10):
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True)

            # Create subdirectories
            (self.save_dir / "images").mkdir(exist_ok=True)
            (self.save_dir / "plots").mkdir(exist_ok=True)
            (self.save_dir / "metrics").mkdir(exist_ok=True)

            # Metrics tracking
            self.g_losses = []
            self.d_losses = []
            self.g_running_avg = []
            self.d_running_avg = []
            self.learning_rates = {'g': [], 'd': []}
            self.quality_metrics = {'healthy_ratio': [], 'mode_collapse_score': []}

            # Training stability metrics
            self.gradient_norms = {'g': [], 'd': []}
            self.loss_ratios = []
            self.log_interval = log_interval

            # Setup logging
            self.setup_logging()

        def setup_logging(self):
            # Configure detailed logging
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(log_format)

            # File handler
            fh = logging.FileHandler(self.save_dir / 'training.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)

            # Stream handler
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(logging.INFO)

            # Get logger
            self.logger = logging.getLogger('GANTraining')
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

        def update(self, g_loss, d_loss, g_lr, d_lr, g_norm=None, d_norm=None):
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            self.learning_rates['g'].append(g_lr)
            self.learning_rates['d'].append(d_lr)

            if g_norm is not None and d_norm is not None:
                self.gradient_norms['g'].append(g_norm)
                self.gradient_norms['d'].append(d_norm)

            # Calculate running averages
            window = min(50, len(self.g_losses))
            if window > 0:
                self.g_running_avg.append(np.mean(self.g_losses[-window:]))
                self.d_running_avg.append(np.mean(self.d_losses[-window:]))

            # Calculate loss ratio
            if d_loss != 0:
                self.loss_ratios.append(g_loss / d_loss)

        def plot_training_progress(self, epoch):
            plt.figure(figsize=(20, 12))

            # Plot 1: Losses
            plt.subplot(2, 2, 1)
            plt.plot(self.g_losses[-1000:], label='Generator Loss', alpha=0.3)
            plt.plot(self.d_losses[-1000:], label='Discriminator Loss', alpha=0.3)
            plt.plot(self.g_running_avg[-1000:], label='G Running Avg', linewidth=2)
            plt.plot(self.d_running_avg[-1000:], label='D Running Avg', linewidth=2)
            plt.title('Training Losses')
            plt.legend()

            # Plot 2: Learning Rates
            plt.subplot(2, 2, 2)
            plt.plot(self.learning_rates['g'], label='Generator LR')
            plt.plot(self.learning_rates['d'], label='Discriminator LR')
            plt.title('Learning Rates')
            plt.legend()

            # Plot 3: Gradient Norms
            plt.subplot(2, 2, 3)
            plt.plot(self.gradient_norms['g'][-1000:], label='Generator Gradients')
            plt.plot(self.gradient_norms['d'][-1000:], label='Discriminator Gradients')
            plt.title('Gradient Norms')
            plt.legend()

            # Plot 4: Loss Ratios
            plt.subplot(2, 2, 4)
            plt.plot(self.loss_ratios[-1000:], label='G/D Loss Ratio')
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            plt.title('Generator/Discriminator Loss Ratio')
            plt.legend()

            plt.tight_layout()
            plt.savefig(self.save_dir / "plots" / f'training_metrics_epoch_{epoch}.png')
            plt.close()

        def check_training_quality(self, g_loss, d_loss):
            """Enhanced training quality checks"""
            issues = []

            # Check recent losses
            recent_g_losses = self.g_losses[-100:] if len(self.g_losses) >= 100 else self.g_losses
            recent_d_losses = self.d_losses[-100:] if len(self.d_losses) >= 100 else self.d_losses

            # Mode collapse detection
            g_loss_std = np.std(recent_g_losses)
            if g_loss_std < 0.01:
                issues.append("WARNING: Possible mode collapse (low G loss variance)")

            # Discriminator dominance
            if np.mean(recent_d_losses) < 0.1:
                issues.append("WARNING: Discriminator too strong")

            # Generator weakness
            if np.mean(recent_g_losses) > 5.0:
                issues.append("WARNING: Generator may be struggling")

            # Loss ratio instability
            if len(self.loss_ratios) > 0:
                recent_ratios = self.loss_ratios[-100:]
                if np.std(recent_ratios) > 2.0:
                    issues.append("WARNING: Unstable G/D loss ratio")

            # Gradient vanishing
            if len(self.gradient_norms['g']) > 0:
                recent_g_grads = self.gradient_norms['g'][-100:]
                if np.mean(recent_g_grads) < 0.01:
                    issues.append("WARNING: Possible vanishing gradients in Generator")

            return issues

        def log_epoch_summary(self, epoch, avg_g_loss, avg_d_loss, g_lr, d_lr):
            """Log comprehensive epoch summary"""
            summary = [
                f"\n{'=' * 50}",
                f"Epoch {epoch} Summary:",
                f"{'=' * 50}",
                f"Average Losses:",
                f"  Generator: {avg_g_loss:.4f}",
                f"  Discriminator: {avg_d_loss:.4f}",
                f"Learning Rates:",
                f"  Generator: {g_lr:.6f}",
                f"  Discriminator: {d_lr:.6f}",
            ]

            if len(self.gradient_norms['g']) > 0:
                summary.extend([
                    f"Gradient Norms (avg last 100):",
                    f"  Generator: {np.mean(self.gradient_norms['g'][-100:]):.4f}",
                    f"  Discriminator: {np.mean(self.gradient_norms['d'][-100:]):.4f}"
                ])

            if len(self.loss_ratios) > 0:
                summary.extend([
                    f"Loss Ratio (G/D):",
                    f"  Current: {self.loss_ratios[-1]:.4f}",
                    f"  Avg (last 100): {np.mean(self.loss_ratios[-100:]):.4f}"
                ])

            # Add training quality issues
            issues = self.check_training_quality(avg_g_loss, avg_d_loss)
            if issues:
                summary.extend([
                    f"Training Issues Detected:",
                    *[f"  - {issue}" for issue in issues]
                ])

            summary.append(f"{'=' * 50}")

            # Log the summary
            self.logger.info('\n'.join(summary))

    def get_gradient_norm(self, model):
        """Calculate gradient norm for a model"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train(self, num_epochs=500, save_interval=10, patience=30):
        """Enhanced training loop with comprehensive monitoring"""
        scaler = torch.amp.GradScaler() if self.use_cuda else None

        best_fid = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0

        # Initialize training monitor
        self.monitor = TrainingMonitor()

        for epoch in range(num_epochs):
            running_d_loss = 0.0
            running_g_loss = 0.0
            batch_count = 0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (real_images, labels) in enumerate(pbar):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)

                # Train step
                d_loss, g_loss = self.train_step(real_images, labels)

                # Calculate gradient norms
                g_norm = self.get_gradient_norm(self.generator)
                d_norm = self.get_gradient_norm(self.discriminator)

                # Get current learning rates
                g_lr = self.g_optimizer.param_groups[0]['lr']
                d_lr = self.d_optimizer.param_groups[0]['lr']

                # Update monitor
                self.monitor.update(g_loss, d_loss, g_lr, d_lr, g_norm, d_norm)

                running_d_loss += d_loss
                running_g_loss += g_loss
                batch_count += 1

                # Update progress bar
                if batch_idx % 10 == 0:
                    issues = self.monitor.check_training_quality(g_loss, d_loss)
                    status = {
                        'D_loss': f"{d_loss:.4f}",
                        'G_loss': f"{g_loss:.4f}",
                        'G_lr': f"{g_lr:.6f}",
                        'D_lr': f"{d_lr:.6f}",
                        'No Improv': epochs_without_improvement
                    }

                    if issues:
                        status['Issues'] = len(issues)

                    if self.use_cuda:
                        status['GPU_mem'] = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"

                    pbar.set_postfix(status)

                # Generate samples periodically
                if batch_idx % 500 == 0:
                    self._generate_sample_images(epoch, batch_idx)

            # Calculate epoch averages
            avg_d_loss = running_d_loss / batch_count
            avg_g_loss = running_g_loss / batch_count

            # Update schedulers
            self.g_scheduler.step(avg_g_loss)
            self.d_scheduler.step(avg_d_loss)

            # Log epoch summary
            self.monitor.log_epoch_summary(
                epoch + 1,
                avg_g_loss,
                avg_d_loss,
                self.g_optimizer.param_groups[0]['lr'],
                self.d_optimizer.param_groups[0]['lr']
            )

            # Plot training progress
            if (epoch + 1) % save_interval == 0:
                self.monitor.plot_training_progress(epoch + 1)
                self.save_checkpoint(epoch + 1)

            # Early stopping check
            current_metric = avg_g_loss  # Or use FID score if available
            if current_metric < best_fid:
                best_fid = current_metric
                epochs_without_improvement = 0
                best_epoch = epoch
                self.save_checkpoint(epoch + 1, is_best=True)
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                self.monitor.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. Best epoch was {best_epoch + 1}")
                break

        # Final save and plotting
        self.monitor.plot_training_progress(num_epochs)
        self.save_checkpoint(num_epochs, is_best=False)
        self.monitor.logger.info("Training completed!")

        # Log training progress
        logger.info(f"Issues: {self.monitor.check_training_quality(avg_g_loss, avg_d_loss)}")
        logger.info(f"Epoch {epoch + 1} - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")

    # Add gradient penalty computation
    def compute_gradient_penalty(self, real_images, fake_images, labels):
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_images)

        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated, labels)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, real_images, labels):
        batch_size = real_images.size(0)

        # Add noise to labels for regularization
        real_labels = torch.ones(batch_size, 1, device=self.device) * self.real_label_val
        fake_labels = torch.zeros(batch_size, 1, device=self.device) * self.fake_label_val

        # Add label noise
        label_noise = 0.05
        real_labels += torch.randn_like(real_labels) * label_noise
        fake_labels += torch.randn_like(fake_labels) * label_noise

        # Train discriminator
        self.d_optimizer.zero_grad()

        # Add noise to real images for regularization
        real_images_noisy = real_images + torch.randn_like(real_images) * 0.05

        # Generate fake images with varied noise
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        noise = noise + torch.randn_like(noise) * 0.05  # Add noise to latent space

        fake_images = self.generator(noise, labels)
        processed_fake = self.background_processor.process_image(fake_images)

        # Use instance noise (reduces to 0 over time)
        instance_noise = 0.05 * (1 - min(self.current_epoch / 50, 1))
        real_images_noisy += torch.randn_like(real_images) * instance_noise
        processed_fake += torch.randn_like(processed_fake) * instance_noise

        d_loss_real = self.criterion(
            self.discriminator(real_images_noisy, labels),
            real_labels
        )
        d_loss_fake = self.criterion(
            self.discriminator(processed_fake.detach(), labels),
            fake_labels
        )

        # Add gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_images, processed_fake.detach(), labels)

        d_loss = (d_loss_real + d_loss_fake) * 0.5 + 10.0 * gradient_penalty
        d_loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()

        # Train generator (less frequently than discriminator)
        if self.current_batch % 2 == 0:  # Train G every other batch
            self.g_optimizer.zero_grad()
            g_loss = self.criterion(
                self.discriminator(processed_fake, labels),
                real_labels  # Use real labels for generator training
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

def main():
    # Configuration
    network_path = r"\\192.168.1.66\plex\hentai\processed_images"
    local_cache_dir = "local_cache"
    batch_size = 32
    num_epochs = 100
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