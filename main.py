import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import shutil
from pathlib import WindowsPath
import time
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkPathHandler:
    def __init__(self, network_path: str, max_retries: int = 3, retry_delay: int = 5):
        self.network_path = Path(network_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def check_path_accessible(self) -> bool:
        """Check if network path is accessible"""
        try:
            return self.network_path.exists()
        except (PermissionError, OSError):
            return False

    def wait_for_access(self) -> bool:
        """Wait for network path to become accessible"""
        for attempt in range(self.max_retries):
            if self.check_path_accessible():
                return True
            logger.warning(f"Network path not accessible, attempt {attempt + 1}/{self.max_retries}")
            time.sleep(self.retry_delay)
        return False


# [Previous imports remain the same...]
import shutil
from pathlib import WindowsPath
import time
from typing import Optional


class NetworkPathHandler:
    def __init__(self, network_path: str, max_retries: int = 3, retry_delay: int = 5):
        self.network_path = Path(network_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def check_path_accessible(self) -> bool:
        """Check if network path is accessible"""
        try:
            return self.network_path.exists()
        except (PermissionError, OSError):
            return False

    def wait_for_access(self) -> bool:
        """Wait for network path to become accessible"""
        for attempt in range(self.max_retries):
            if self.check_path_accessible():
                return True
            logger.warning(f"Network path not accessible, attempt {attempt + 1}/{self.max_retries}")
            time.sleep(self.retry_delay)
        return False


class AnimeDataset(Dataset):
    def __init__(self, network_path: str, image_size: int = 256):
        self.network_handler = NetworkPathHandler(network_path)
        if not self.network_handler.wait_for_access():
            raise RuntimeError(f"Could not access network path: {network_path}")

        self.root_dir = Path(network_path)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        logger.info("Scanning network directory for images...")
        self._scan_directories()

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

            logger.info(f"Found {len(self.image_paths)} images across {len(self.label_to_idx)} categories")

        except Exception as e:
            logger.error(f"Error scanning network directory: {e}")
            raise RuntimeError("Failed to scan network directory")

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                label = self.labels[idx]
                return image, label
            except (OSError, PermissionError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load image after {max_retries} attempts: {img_path}")
                    # Return a random valid index instead
                    return self.__getitem__(np.random.randint(self.__len__()))

class TrainingMonitor:
    def __init__(self, save_dir="training_progress"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.g_losses = []
        self.d_losses = []
        self.fid_scores = []  # We'll track FID if available
        self.inception_scores = []  # We'll track IS if available

    def update(self, g_loss, d_loss):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)

    def plot_losses(self, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.title('Training Losses Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir / f'losses_epoch_{epoch}.png')
        plt.close()

    def save_samples(self, images, epoch, prefix='generated'):
        # Save a grid of images
        grid = make_grid(images, normalize=True)
        save_image(grid, self.save_dir / f'{prefix}_samples_epoch_{epoch}.png')

    def check_training_quality(self, g_loss, d_loss):
        """
        Check if training is progressing well based on loss patterns
        """
        # Check for common training issues
        issues = []

        # Check if generator is learning
        if len(self.g_losses) > 100:  # Wait for some training history
            recent_g_loss = np.mean(self.g_losses[-20:])
            if recent_g_loss > np.mean(self.g_losses[:20]):
                issues.append("Generator loss is increasing")

            # Check for mode collapse
            if np.std(self.g_losses[-20:]) < 0.01:
                issues.append("Possible mode collapse detected")

            # Check discriminator dominance
            if np.mean(self.d_losses[-20:]) < 0.1:
                issues.append("Discriminator may be too strong")

            # Check vanishing gradients
            if abs(recent_g_loss - self.g_losses[-2]) < 1e-7:
                issues.append("Possible vanishing gradients")

        return issues

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 50)

        self.model = nn.Sequential(
            # Input: latent_dim + 50 (embedded label)
            nn.Linear(latent_dim + 50, 256 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([z, label_embedding], dim=1)
        return self.model(x)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None

        # Initialize dataset with network path
        logger.info(f"Initializing dataset from network path: {network_path}")
        self.dataset = AnimeDataset(network_path, image_size)

        # Create data loader with appropriate num_workers
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if os.name == 'nt' else 4,  # Windows compatibility
            pin_memory=True
            )

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

    def train(self, num_epochs=100, save_interval=5):
        best_fid = float('inf')  # Track best model performance

        for epoch in range(num_epochs):
            running_d_loss = 0.0
            running_g_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for real_images, labels in pbar:
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)

                # ======= Train Discriminator =======
                self.d_optimizer.zero_grad()

                # Real images
                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)

                output_real = self.discriminator(real_images, labels)
                d_loss_real = self.criterion(output_real, label_real)

                # Fake images
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(noise, labels)
                output_fake = self.discriminator(fake_images.detach(), labels)
                d_loss_fake = self.criterion(output_fake, label_fake)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # ======= Train Generator =======
                self.g_optimizer.zero_grad()

                # Try to fool the discriminator
                output_fake = self.discriminator(fake_images, labels)
                g_loss = self.criterion(output_fake, label_real)

                g_loss.backward()
                self.g_optimizer.step()

                # Update running losses
                running_d_loss += d_loss.item()
                running_g_loss += g_loss.item()

                # Update monitor
                self.monitor.update(g_loss.item(), d_loss.item())

                # Check training quality
                issues = self.monitor.check_training_quality(g_loss.item(), d_loss.item())
                if issues:
                    for issue in issues:
                        logger.warning(f"Training issue detected: {issue}")

                pbar.set_postfix({
                    'D_loss': f"{d_loss.item():.4f}",
                    'G_loss': f"{g_loss.item():.4f}"
                })

            # Generate and save sample images
            if (epoch + 1) % save_interval == 0:
                with torch.no_grad():
                    # Generate samples for different characters
                    num_samples = min(4, len(self.dataset.label_to_idx))
                    sample_labels = torch.arange(num_samples).to(self.device)
                    fake_images = self.generate_samples(num_samples, sample_labels)
                    self.monitor.save_samples(fake_images, epoch + 1)

                    # Plot and save loss curves
                    self.monitor.plot_losses(epoch + 1)

                    # Save checkpoint
                    self.save_checkpoint(epoch + 1)

            avg_d_loss = running_d_loss / len(self.dataloader)
            avg_g_loss = running_g_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1} - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")


def main():
    # Configuration
    network_path = r"\\192.168.1.66\plex\anime"
    local_cache_dir = "local_cache"  # Optional local cache directory
    batch_size = 32
    num_epochs = 100
    save_interval = 5
    image_size = 128
    latent_dim = 100

    # Initialize and train
    trainer = AnimeGeneratorTrainer(
        network_path=network_path,
        batch_size=batch_size,
        latent_dim=latent_dim,
        image_size=image_size,
        local_cache_dir=local_cache_dir
    )

    # Start training
    trainer.train(num_epochs=num_epochs, save_interval=save_interval)

    # After training, generate samples
    logger.info("Training completed. Generating sample images...")
    trainer.load_checkpoint("checkpoints/best_model.pt")

    # Generate samples for different characters
    num_samples = min(16, len(trainer.dataset.label_to_idx))
    sample_labels = torch.arange(num_samples).to(trainer.device)
    generated_images = trainer.generate_samples(num_samples, sample_labels)

    # Save final samples
    trainer.monitor.save_samples(generated_images, epoch=num_epochs, prefix='final')
    logger.info(f"Final samples saved to {trainer.monitor.save_dir}")


if __name__ == "__main__":
    main()