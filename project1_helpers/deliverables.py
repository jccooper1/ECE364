import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision 
from torchvision import transforms
import numpy as np
import os
import torch.optim as optim
from PIL import Image
from iou import *
from tqdm import tqdm 

class BFSDataset(Dataset):
    def __init__(self, root_dir, scene_name, split='train', transform_size=(1430, 950)):
        self.transform_size = transform_size
        self.samples = []
        
        # Get paths
        self.image_dir = os.path.join(root_dir, "Images", scene_name)
        self.annotation_dir = os.path.join(root_dir, "Annotations", scene_name)
        
        # Get all dates
        for date in os.listdir(self.image_dir):
            # Skip hidden files
            if date.startswith('.'):
                continue
                
            date_path = os.path.join(self.image_dir, date)
            # Make sure it's a directory
            if not os.path.isdir(date_path):
                continue
                
            # Get all timestamps in this date folder
            for timestamp in os.listdir(date_path):
                if timestamp.startswith('.'):
                    continue
                    
                if timestamp.endswith('.jpg'):
                    # Construct full paths
                    image_path = os.path.join(self.image_dir, date, timestamp)
                    ann_path = os.path.join(self.annotation_dir, date, 
                                          timestamp.replace('.jpg', '.png'))
                    
                    if os.path.exists(ann_path):
                        self.samples.append((image_path, ann_path))
        
        # Sort samples for reproducibility
        self.samples.sort()
        
        # Split into train/val
        n_total = len(self.samples)
        n_train = n_total // 2
        
        if split == 'train':
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]
            
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, ann_path = self.samples[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Load and transform annotation
        annotation = Image.open(ann_path).convert('L')  # grayscale
        annotation = self.transform(annotation)
        annotation = (annotation > 0.5).float()  # binarize
        
        return image, annotation

class SegmentationModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        # Changed from super().__init__(self) to super().__init__()
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )
        
        self.output = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return torch.sigmoid(x)

def training_loop(model, criterion, optimizer, n_epochs, train_loader, val_loader):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    metrics = {'loss_values': [], 'train_ious': [], 'val_ious': []}
    
    for n in tqdm(range(n_epochs)):
        epoch_loss = 0
        epoch_iou = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_iou += iou(y_pred, y_batch)
            
        metrics['loss_values'].append(epoch_loss/len(train_loader))
        metrics['train_ious'].append(epoch_iou/len(train_loader))
        
        val_iou = 0
        for x_batch, y_batch in val_loader:
            with torch.no_grad():
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                val_iou += iou(y_pred, y_batch)
        metrics['val_ious'].append(val_iou/len(val_loader))
    
    return model, metrics

# Example usage
if __name__ == "__main__":
    # Example for one scene
    scene_name = "Buffalo Grove at Deerfield East"
    root_dir = '.\\BFSData'  # Current directory
    
    # Create datasets
    train_dataset = BFSDataset(root_dir, scene_name, split='train')
    val_dataset = BFSDataset(root_dir, scene_name, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model, criterion, and optimizer
    model = SegmentationModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    trained_model, metrics = training_loop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=50,
        train_loader=train_loader,
        val_loader=val_loader
    )