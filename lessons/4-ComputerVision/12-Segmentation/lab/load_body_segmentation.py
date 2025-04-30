import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

class BodySegmentationDataset(Dataset):
    def __init__(self, dataset_path, train=True, train_ratio=0.8, transform=None, mask_transform=None):
        """
        Body segmentation dataset loader
        
        Args:
            dataset_path: Root directory of the dataset containing 'images' and 'masks' folders
            train: If True, load the training set, otherwise load the validation set
            train_ratio: Ratio of data to use for training
            transform: Transforms to apply to images
            mask_transform: Transforms to apply to masks
        """
        self.img_path = os.path.join(dataset_path, 'images')
        self.mask_path = os.path.join(dataset_path, 'masks')
        
        # Get all filenames
        self.fnames = sorted(os.listdir(self.img_path))
        
        # Split into training and validation
        split_idx = int(len(self.fnames) * train_ratio)
        if train:
            self.fnames = self.fnames[:split_idx]
        else:
            self.fnames = self.fnames[split_idx:]
            
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((384, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        if self.mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((384, 512), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.fnames[idx]
        img_path = os.path.join(self.img_path, img_name)
        mask_path = os.path.join(self.mask_path, img_name)  # Assuming mask has same filename
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask
    
    def get_all_samples(self):
        """
        Get all images and masks at once
        
        Returns:
            images: tensor of shape [num_samples, channels, height, width]
            masks: tensor of shape [num_samples, 1, height, width]
        """
        images = []
        masks = []
        
        for i in range(len(self)):
            image, mask = self[i]
            images.append(image)
            masks.append(mask)
            
        # Stack to create batched tensors
        return torch.stack(images), torch.stack(masks)

def create_dataloaders(dataset_path, train_batch_size=8, val_batch_size=4):
    """
    Create training and validation dataloaders
    
    Args:
        dataset_path: Root directory of the dataset
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        
    Returns:
        train_dataloader, val_dataloader
    """
    # Create datasets
    train_dataset = BodySegmentationDataset(dataset_path, train=True)
    val_dataset = BodySegmentationDataset(dataset_path, train=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_dataloader, val_dataloader

def visualize_sample(dataset, idx=0, figsize=(12, 6)):
    """
    Visualize a sample from the dataset
    
    Args:
        dataset: Dataset object
        idx: Index of the sample to visualize
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    # Get sample
    image, mask = dataset[idx]
    
    # Convert from tensor to numpy for visualization
    image = image.permute(1, 2, 0).numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    mask = mask.squeeze().numpy()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title('Image')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show() 