import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia as K

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn(y_pred, y_true):
    """
    Compute the L2 loss between the predicted and ground truth 4-point homography.

    Parameters:
    predicted_delta (torch.Tensor): The predicted perturbations of the corners by the network.
    img_a (torch.Tensor): Original images.
    patch_b (torch.Tensor): Patches from the warped images.
    corners (torch.Tensor): Ground truth corner coordinates.

    Returns:
    torch.Tensor: The loss value.
    """
    # criterion = nn.MSELoss()
    
    return torch.sqrt(torch.sum((y_pred - y_true)**2, dim=1, keepdim=True))

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net()

    def forward(self, a):
        return self.model(a)

    def training_step(self, batch, batch_idx):
        img_batch, corners= batch
        delta = self.model(img_batch)
        loss = LossFn(delta, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_batch, corners= batch
        delta = self.model(img_batch)
        loss = LossFn(delta, corners)
        return {"val_loss": loss}

    def validation_epoch_end(outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(Net, self).__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2d(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.maxpool2d(x)
        x = x.reshape(-1, 128 * 16 * 16)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        out = self.fc2(x)
        return out