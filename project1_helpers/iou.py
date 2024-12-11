import torch

def iou(model, dataloader, device='cpu'):
    '''
    model: Instance of a PyTorch nn.Module class with a defined forward() function that takes images as inputs
    dataloader: PyTorch DataLoader object that represents the dataset we will compute IoU over
    device (optional): string naming the device for computation, only needs to be changed for GPU training/evaluation
    '''
    intersection_pixel_count = 0
    union_pixel_count = 0
    with torch.no_grad():
        for images, targets in dataloader:
            # images should be shape (N, 3, H, W) for N images in a batch
            # targets should be shape (N, 1, H, W) for N images in a batch
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            # predictions may be shape (N, 1, H, W) or (N, 2, H, W) depending on choice of implementation and loss function
            if predictions.size(1) == 1:
                # single prediction channel, result from model already passed through sigmoid
                binary_predictions = predictions > 0.5 # set pixels with >0.5 probability to one
            else:
                # two prediction channels, index 1 represents positive/foreground class, need to pass through softmax first
                prediction_probabilities = torch.softmax(predictions, dim=1)
                binary_predictions = prediction_probabilities[:, 1] > 0.5
                binary_predictions = binary_predictions.unsqueeze(1) # return to shape (N, 1, H, W)
            # add to intersection and union counts
            intersection_pixel_count += torch.sum((binary_predictions*targets)>0).item()
            union_pixel_count += torch.sum((binary_predictions+targets)>0).item()
    # return IoU over the given dataset
    return intersection_pixel_count/union_pixel_count
