import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision 
from torchvision import transforms
import numpy as np
import os
import torch.optim as optim
from PIL import Image


class BFSDataset(Dataset):
    def __init__(self,root_dir,scene_name, split= 'train', transform_size=(360,240)):
        # possibly add arguments to the constructor
        # fill in code to load all the images and annotations and assign them as attributes to the dataset
        #now we initialize the dataset 
        #we train the whole dataset and the split its half for train 
        self.transform_size = transform_size
        #use this to store the train samples 
        self.samples = []
        self.image_dir= os.path.join(root_dir,"Images",scene_name)
        self.annotation_dir= os.path.join(root_dir,"Annotations",scene_name)
        #firstly we get the path of each picture and annotation
        for date in os.listdir(self.image_dir):
            date_dir= os.path.join(self.image_dir,date)
            for timestamp in os.listdir(date_dir):
                #since the image part is jpg and annonation part is png, we need to make sure that they are the same type 
                if(timestamp.endswith('.jpg')):
                    image_path= os.path.join(self.image_dir,date_dir,timestamp)
                    #we change the type all into png
                    ann_path= os.path.join(self.annotation_dir,date_dir,timestamp[:-4]+'.png')
                    if(ann_path.os.path.exists()):
                        #if the annotation exists, we append the path to the samples
                        self.samples.append((image_path,ann_path))
        #we sort the samples based on the name but it may be abundant
        self.samples.sort()
        n_total = len(self.samples)
        n_half = n_total//2
        #we split the dataset into two parts specifically for train and validation test
        if(split == 'train'):
            self.samples = self.samples[:n_half]
        else:
            self.samples = self.samples[n_half:]
        #reduce the work time reduce the its size to half to transform the image 
        self.transform = transforms.Compose([transforms.Resize(self.transform_size),transforms.ToTensor()])
    def __len__(self):
        # return length of the dataset, i.e. number of (image, annotation) pairs
        return len(self.samples)
    def __getitem(self, idx):
        # return one image, annotation pair
        # fill in code to access an (image, annotation) pair in the dataset
        #find the path to the the data we want 
        image_path, ann_path = self.samples[idx]
        #open the image then transform them into tensor
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        #open the annotation and transform it into tensor
        annotation = Image.open(ann_path).Grayscale()  # convert to grayscale
        annotation = self.transform(annotation)
        #not sure here just for test
        #annotation = (annotation>0.5).float()

        return image, annotation

class SegmentationModel(nn.Module):
    #since the input is RGB image, the input channel is 3
    #since the output is the probability of the foreground, the output channel is 1
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__(self)
        # possibly add arguments to the constructor
        # fill in code to declare attributes and model layers
        #we use the autoencoder to train the model with 3 layers
        self.encoder = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1,stride=2),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=2),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1,stride=2),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2),
                                        nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1,stride=2),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2))
        self.output = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
    def forward(self, x):
        # fill in forward function

        # your forward function will return either an (N, 1, H, W) tensor or (N, 2, H, W) depending on
        # if you implementation has two classes (background, foreground) for output channels or just
        # one class, i.e. probability of foreground
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x.sigmoid()

def training_loop(model, train_dataloader, validation_dataloader, optimizer, criterion):
    # example arguments for training loop, feel free to create your own

    # return your trained model and any metrics you computed and would like for plotting, displaying in your report
    return model, metrics
