#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import csv
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel, UnsupHomographyModel
from Network.Network import *
from Network.Network import LossFn
import cv2
import sys
import os
import numpy as np
import pandas as pd
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import zipfile
from Misc.DataUtils import SetupAll


#CHANGED 

def getimagename(path):
    """
    Get the image name from the path
    """
    path = '../Data/Train_generated/patchA'
    img_name = []
    for img in os.listdir(path):
        img_name.append(img)
    return img_name

def GenerateBatch(BasePath, MiniBatchSize, ModelType, path_patchA, path_patchB):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    image_batch = []
    labels_batch = []

    path_patchA = BasePath + 'Train_generated' + '/' + 'patchA' + '/'
    path_patchB = BasePath + 'Train_generated' + '/' + 'patchB' + '/'
    coords = BasePath + 'Train_generated' + '/' + 'TrainH4.npy'

    CoordinatesBatch = torch.from_numpy(np.load(coords))
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        img_name = getimagename(path_patchA)
        RandIdx = random.randint(0, len(img_name) - 1)
        path_patchA= path_patchA + img_name[RandIdx] 
        path_patchB= path_patchB + img_name[RandIdx] 
        ImageNum +=1

            ##########################################################
            # Add any standardization or data augmentation here!
            ##########################################################
        imageA = cv2.imread(path_patchA, cv2.IMREAD_GRAYSCALE)
        imageB = cv2.imread(path_patchB, cv2.IMREAD_GRAYSCALE)

        label = CoordinatesBatch[RandIdx]

        imageA = torch.from_numpy((imageA.astype(np.float32) / 255.0))
        imageB = torch.from_numpy((imageB.astype(np.float32) / 255.0))

        label = label.to(torch.float32) 

        img = torch.stack((imageA, imageB), dim=0)

        # Append All Images and Mask
        image_batch.append(img.to('cpu'))
        labels_batch.append(label.to('cpu'))
        
        return torch.stack(image_batch), torch.stack(labels_batch)

def patchindex(corner):
    """
    Get the patch indices
    """
    patch_indices = []
    for i in range(len(corner)):
        x = corner[i][0]
        y = corner[i][1]
        patch_indices.append([x, y])
    return patch_indices

#Load data for unsupervised loop
def unsupdata(BasePath, dir, points_list, batch_size, shuffle = True):

    patchPair = []
    cor1 = []
    patche2 = []
    imag1 = []


    if(len(dir) < batch_size):
        print("The data has only ", len(dir) , " images and you are trying to get ",batch_size, " images")
        return 0

    for n in range(batch_size):
        index = random.randint(0, len(dir)-1)
       
        patch1 = BasePath + "Train_generated/patchA/" + str(index) + '.jpg'
        patch1 = cv2.imread(patch1, cv2.IMREAD_GRAYSCALE)

        patch2 = BasePath + "Train_generated/patchB/" + str(index) + '.jpg'
        patch2 = cv2.imread(patch2, cv2.IMREAD_GRAYSCALE)

        image1 = BasePath + "Train/" + str(index) + '.jpg'
        image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE) 

        if(patch1 is None) or (patch2 is None):
            print(patch1, " is empty. Ignoring ...")
            continue

        patch1 = np.float32(patch1) 
        patch2 = np.float32(patch2) 
        image1 = np.float32(image1)
        
        #combine images along depth
        patch_pair = np.dstack((patch1, patch2))     
        corner1 = points_list[index, 0:4]
        
        
        patchPair.append(patch_pair)
        cor1.append(corner1)
        patche2.append(patch2.reshape(128, 128, 1))

    
        imag1.append(image1)

    patch_indices = patchindex(np.array(cor1))    
    return np.array(patchPair), np.array(cor1), np.array(patche2), np.array(imag1), patch_indices



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

def TrainOperation(DirNamesTrain,TrainCoordinates,NumTrainSamples,ImageSize,NumEpochs,MiniBatchSize,SaveCheckPoint,
    CheckPointPath,DivTrain,LatestFile,BasePath,LogsPath,ModelType,):
    model = HomographyModel().to('cpu')


    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=0.0001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    ########################################################
    # Load latest Checkpoint from the Checkpoint Directory #
    ########################################################

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")



    ############################
    ###### TRAINING LOOP #######
    ############################
    path_patchA = BasePath + os.sep + 'Train_generated' + os.sep + 'patchA'
    path_patchB = BasePath + os.sep + 'Train_generated' + os.sep + 'patchB'

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        print("Number of Iterations per Epoch " + str(NumIterationsPerEpoch))

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, H4PtBatch = GenerateBatch(BasePath, MiniBatchSize, ModelType, path_patchA, path_patchB)

            
            # Call training step directly (usually done by PyTorch Lightning's Trainer)
            batch = (I1Batch, H4PtBatch)

            # Call the training_step with the constructed batch and an arbitrary batch index (e.g., 0)
            model.train()
            loss_dict = model.training_step(batch,0)
            loss = loss_dict['loss']
            
            ##########################################################
            ################## Backpropagation #######################
            ##########################################################

            # Compute the loss between the predicted and true H4Pt
            # Forward pass through the model to get the predicted delta
            PredictedH4PtBatch = model(I1Batch)

            LossThisBatch = LossFn(PredictedH4PtBatch, H4PtBatch)
            
            Optimizer.zero_grad()
            LossThisBatch.sum().backward()
            Optimizer.step()

            # Print Loss
            print(
                "Epoch: " + str(Epochs) + " Iteration: " + str(PerEpochCounter) + " Loss: " + str(LossThisBatch.sum().item())
            )

            #########################################################
            # Save checkpoint every some SaveCheckPoint's iterations
            #########################################################
            # Save the Model learnt in this epoch
        SaveName = (
            CheckPointPath
            + str(Epochs)
            + "model.ckpt")
            

        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
            )
        print("\n" + SaveName + " Model Saved...")

            
        Batch = (I1Batch, H4PtBatch)  # Batch is a tuple of inputs and targets
        model.eval()
        result = model.validation_step(Batch, 0)  # Forward pass
            
            
            #####################
            #### Tensorboard ####
            #####################

        Writer.add_scalar(
            "LossEveryEpoch_val",
            result["val_loss"].mean(),
            Epochs,
        )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

#UNSUPERVISED TRAINING LOOP
def TrainModel(PatchPairsPH, CornerPH, Patch2PH, Image1PH, patchIndicesPH, DirNamesTrain, CornersTrain, NumTrainSamples, ImageSize, NumEpochs, BatchSize, SaveCheckPoint, CheckPointPath, LatestFile, BasePath, LogsPath):
    print("Training Unsupervised model....")

    # Assuming unsupervised_HomographyNet is a PyTorch model
    model = UnsupHomographyModel()  

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Tensorboard
    writer = SummaryWriter(log_dir=LogsPath)

    AccOverEpochs = torch.tensor([0, 0])

    if LatestFile is not None:
        # Load the latest checkpoint
        checkpoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        StartEpoch = checkpoint['epoch']
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')

    lossL1 = []
    for epoch in tqdm(range(StartEpoch, NumEpochs)):

        NumIterationsPerEpoch = int(NumTrainSamples / BatchSize)
        Loss = []
        epoch_loss = 0

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Assuming loadData returns PyTorch tensors
            PatchPairsBatch, Corner1Batch, patch2Batch, Image1Batch, patchIndicesBatch = unsupdata(BasePath, DirNamesTrain, CornersTrain, BatchSize, shuffle=True)

            batch_unsup = (PatchPairsBatch, Corner1Batch, patch2Batch, Image1Batch, patchIndicesBatch)
            model.train()
            loss_dict = model.training_step(batch_unsup,0)
            loss = loss_dict['loss']
        
            lossL1.append(loss.item())
            epoch_loss += loss.item()

            optimizer.zero_grad()

            # Backward pass and optimization
            
            loss.backward()
            optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            # if PerEpochCounter % SaveCheckPoint == 0:
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss,
            #     }, CheckPointPath + str(epoch) + 'a' + str(PerEpochCounter) + 'model.pt')
            #     print('\n' + SaveName + ' Model Saved...')

        # Tensorboard
        writer.add_scalar('LossEveryIter', np.mean(Loss), epoch * NumIterationsPerEpoch + PerEpochCounter)
        epoch_loss /= NumIterationsPerEpoch

        print("Printing Epoch:  ", np.mean(Loss), "\n")
        lossL1.append(np.mean(Loss))

        # Save model every epoch
        SaveName = CheckPointPath + str(epoch) + 'unsupmodel.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, SaveName)

        model.eval()
        result = model.validation_step(batch_unsup, 0)  # Forward pass

        writer.add_scalar(
            "LossEveryEpoch_val",
            result["val_loss"].mean(),
            epoch,
        )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
        writer.flush()

    np.savetxt(LogsPath + "loss_unsupervised.txt", np.array(lossL1), delimiter=",")
#######################
# Visualize the batch #
#######################
        
def visualizeBatch(I1Batch, I2Batch, H4PtBatch, patch_size):
    for i in range(len(I1Batch)):
        plt.figure(figsize=(12, 4))

        # Convert the tensor to a numpy array and ensure correct shape
        original_img = I1Batch[i].cpu().numpy()
        if original_img.max() > 1.0:
            original_img = original_img / 255.0  # Normalize to [0, 1] if max value is greater than 1

        warped_img = I2Batch[i].cpu().numpy()
        if warped_img.max() > 1.0:
            warped_img = warped_img / 255.0  # Normalize to [0, 1] if max value is greater than 1

        # Visualize Original Patch
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Patch")

        # Visualize Warped Patch
        plt.subplot(1, 3, 2)
        plt.imshow(warped_img)
        plt.title("Warped Patch")

        # Visualize Perturbations
        plt.subplot(1, 3, 3)
        plt.imshow(original_img)
        for j in range(4):
            dx, dy = H4PtBatch[i][j].cpu().numpy()
            x, y = [0, patch_size[1], 0, patch_size[1]][j], [0, 0, patch_size[0], patch_size[0]][j]  # Patch corner coordinates
            plt.arrow(x, y, dx, dy, head_width=3, head_length=5, fc='red', ec='red')
        plt.title("Perturbations")

        plt.show()

#######################
#### Main Function ####
#######################

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default='../Data/',
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="supervised",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=10,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=32,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="../Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    

    # Visualize the batch
    #visualizeBatch(I1Batch, I2Batch, H4PtBatch, patch_size)



    #setup all the data
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    if Args.ModelType == "supervised":
        TrainOperation(
            DirNamesTrain,
            TrainCoordinates,
            NumTrainSamples,
            ImageSize,
            NumEpochs,
            MiniBatchSize,
            SaveCheckPoint,
            CheckPointPath,
            DivTrain,
            LatestFile,
            BasePath,
            LogsPath,
            ModelType="supervised",  # Replace "supervised" with the desired model type
        )
    else: 

        MiniBatchSize = 64
        CornerPH = torch.FloatTensor(MiniBatchSize, 4, 2)
        PatchPairsPH = torch.FloatTensor(MiniBatchSize, 128, 128, 2)
        Patch2PH = torch.FloatTensor(MiniBatchSize, 128, 128, 1)
        Image1PH = torch.FloatTensor(MiniBatchSize, 240, 320, 1)
        patchIndicesPH = torch.LongTensor(MiniBatchSize, 128, 128, 2)
        CornersTrain = np.load(BasePath + '/Train_generated/TrainH4.npy')

        TrainModel(
            PatchPairsPH, 
            CornerPH, 
            Patch2PH, 
            Image1PH, 
            patchIndicesPH, 
            DirNamesTrain, 
            CornersTrain, 
            NumTrainSamples, 
            ImageSize, 
            NumEpochs, 
            MiniBatchSize, 
            SaveCheckPoint, 
            CheckPointPath, 
            LatestFile, 
            BasePath, 
            LogsPath
        )


if __name__ == "__main__":
    main()
