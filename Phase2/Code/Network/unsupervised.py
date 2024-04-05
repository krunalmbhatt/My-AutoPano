import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia.geometry.transform as K
from Network.supervised import HomographyModel

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn(predicted_patch, patch ):
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
    loss = torch.sqrt(torch.sum((predicted_patch - patch)**2, dim=1, keepdim=True))
    return loss

class UnsupHomographyModel(pl.LightningModule):
    def __init__(self):
        super(UnsupHomographyModel, self).__init__()
        self.model = unsupervisedNet()

    def forward(self, patch_batches, corners_a, patch_b, image_a, patch_indices):
        return self.model(patch_batches, corners_a, patch_b, image_a, patch_indices)

    def training_step(self, batch, batch_idx):
        patch_batches, corners_a, patch_b, image_a, patch_indices = batch
        warped_Pa, patch_b_pred, H_batches = self(patch_batches, corners_a, patch_b, image_a, patch_indices)
        loss = LossFn(patch_b_pred, patch_b)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        patch_batches, corners_a, patch_b, image_a, patch_indices = batch
        warped_Pa, patch_b_pred, H_batches = self(patch_batches, corners_a, patch_b, image_a, patch_indices)
        loss = LossFn(patch_b_pred, patch_b)
        return {"val_loss": loss}

    def validation_epoch_end(outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class unsupervisedNet(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(unsupervisedNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*32*32, 1024)
        self.fc2 = nn.Linear(1024, 8)
        self.dropout = nn.Dropout(0.5)     

    def forward(self, patch_batches, corners_a, patch_b, image_a, patch_indices):
        batch_size, _, h, w = image_a.size()
        x = patch_batches
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2d(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))

        x = self.maxpool2d(x)

        x = x.reshape(batch_size, -1)  # Flatten the tensor
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        H4_batches = self.fc2(x)
        corners_a = corners_a.view(batch_size, 8)
        H_batches = tensor_dlt(H4_batches, corners_a, batch_size)

        M = torch.FloatTensor([[w / 2.0, 0., w / 2.0],
                               [0., h / 2.0, h / 2.0],
                               [0., 0., 1.]])

        M_batches = M.unsqueeze(0).expand(batch_size, -1, -1)
        M_inv_batches = torch.inverse(M).unsqueeze(0).expand(batch_size, -1, -1)

        H_scaled = torch.bmm(torch.bmm(M_inv_batches, H_batches), M_batches)
        warped_Pa = K.warp_perspective(patch_batches[:,1,:,:].unsqueeze(1), H_scaled, dsize=(128,128))
        return warped_Pa, patch_b, H_batches
    
'''
auxiliaryMatrices to build the A matrix in Tensor DLT  
taken from :  https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018/blob/master/code/utils/utils.py
'''

Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)

Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)

Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)



Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)

Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)

Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)

Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)

Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)
Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


def tensor_dlt(H4, corners_a, batch_size):
    corners_a = corners_a.unsqueeze(2)  # batch_size x 8 x 1

    # Solve for H using DLT
    pred_h4p_tile = H4.unsqueeze(2)  # batch_size x 8 x 1
    # 4 points on the second image
    pred_corners_b_tile = pred_h4p_tile + corners_a

    # obtain 8 auxiliary tensors -> expand dimensions by 1 at first -> create batch_size number of copies
    tensor_aux_M1 = torch.tensor(Aux_M1, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M1 = tensor_aux_M1.repeat(batch_size,1,1)

    tensor_aux_M2 = torch.tensor(Aux_M2, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M2 = tensor_aux_M2.repeat(batch_size,1,1)

    tensor_aux_M3 = torch.tensor(Aux_M3, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M3 = tensor_aux_M3.repeat(batch_size,1,1)

    tensor_aux_M4 = torch.tensor(Aux_M4, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M4 = tensor_aux_M4.repeat(batch_size,1,1)

    tensor_aux_M5 = torch.tensor(Aux_M5, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M5 = tensor_aux_M5.repeat(batch_size,1,1)

    tensor_aux_M6 = torch.tensor(Aux_M6, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M6 = tensor_aux_M6.repeat(batch_size,1,1)

    tensor_aux_M71 = torch.tensor(Aux_M71, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M71 = tensor_aux_M71.repeat(batch_size,1,1)

    tensor_aux_M72 = torch.tensor(Aux_M72, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M72 = tensor_aux_M72.repeat(batch_size,1,1)
    
    tensor_aux_M8 = torch.tensor(Aux_M8, dtype=torch.float32).unsqueeze(0)
    tensor_aux_M8 = tensor_aux_M8.repeat(batch_size,1,1)
    
    tensor_aux_Mb = torch.tensor(Aux_Mb, dtype=torch.float32).unsqueeze(0)
    tensor_aux_Mb = tensor_aux_Mb.repeat(batch_size,1,1)

    # Form the equations Ax = b to compute H
    # Build A matrix
    A1 = torch.matmul(tensor_aux_M1, corners_a)  
    A2 = torch.matmul(tensor_aux_M2, corners_a)  
    A3 = tensor_aux_M3  
    A4 = torch.matmul(tensor_aux_M4, corners_a)  
    A5 = torch.matmul(tensor_aux_M5, corners_a) 
    A6 = tensor_aux_M6  
    A7 = torch.matmul(tensor_aux_M71, pred_corners_b_tile) * torch.matmul(tensor_aux_M72, corners_a)  # Column 7
    A8 = torch.matmul(tensor_aux_M71, pred_corners_b_tile) * torch.matmul(tensor_aux_M8, corners_a)  # Column 8


    # reshape A1-A8 as 8x1 and stack them column-wise
    A = torch.stack([A1.view(-1, 8), A2.view(-1, 8), A3.view(-1, 8), A4.view(-1, 8),
                    A5.view(-1, 8), A6.view(-1, 8), A7.view(-1, 8), A8.view(-1, 8)], dim=1)
    A = A.permute(0, 2, 1)

    # Build b matrix
    b = torch.matmul(tensor_aux_Mb, pred_corners_b_tile)

    # Solve the Ax = b to get h11 - h32 as H8 matrix
    H_8 = torch.linalg.solve(A, b)  # batch_size x 8 - has values from H11-H32

    # Add h33 = ones to the last cols to complete H matrix
    h_33 = torch.ones([batch_size, 1, 1], dtype=torch.float32)
    H_9 = torch.cat([H_8, h_33], dim=1)
    H_flat = H_9.view(-1, 9)
    H = H_flat.view(-1, 3, 3)  # batch_size x 3 x 3



    return H

if __name__ == "__main__":
    model = UnsupHomographyModel()
    MiniBatchSize = 64
    CornerPH = torch.randn(MiniBatchSize, 4, 2)
    PatchPairsPH = torch.randn(MiniBatchSize, 2, 128, 128)
    Patch2PH = torch.randn(MiniBatchSize, 1,128,128)
    Image1PH = torch.randn(MiniBatchSize, 1,240,320)
    patchIndicesPH = torch.randn(MiniBatchSize, 2, 128, 128)
    # CornersTrain = np.load('../../Data/Train_generated/TrainH4.npy')

    out = model(PatchPairsPH, CornerPH, Patch2PH, Image1PH, patchIndicesPH)
    print(out.shape)
