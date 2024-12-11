import torch
import torch.nn as nn

from torch.utils.data import Dataset

class BFSDataset(Dataset):
    def __init__(self):
        # possibly add arguments to the constructor

        # fill in code to load all the images and annotations and assign them as attributes to the dataset

    def __len__(self):
        # return length of the dataset, i.e. number of (image, annotation) pairs
        return 

    def __getitem(self, idx):
        # return one image, annotation pair

        # fill in code to access an (image, annotation) pair in the dataset

        return image, annotation

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__(self)
        # possibly add arguments to the constructor

        # fill in code to declare attributes and model layers

    def forward(self, x):
        # fill in forward function

        # your forward function will return either an (N, 1, H, W) tensor or (N, 2, H, W) depending on
        # if you implementation has two classes (background, foreground) for output channels or just
        # one class, i.e. probability of foreground

        return

def training_loop(model, train_dataloader, validation_dataloader, optimizer, criterion):
    # example arguments for training loop, feel free to create your own

    # return your trained model and any metrics you computed and would like for plotting, displaying in your report
    return model, metrics
