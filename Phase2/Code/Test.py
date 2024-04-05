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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import *
from Network.supervised import *
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import pandas as pd
import torch



# Don't generate pyc codes
sys.dont_write_bytecode = True
# def unsupervised_test(BasePath, ModelPath, path_to_save):
#     """
#     Inputs:
#     BasePath: Path to load images from
#     ModelPath: Path to load latest model from
#     path_to_save: Path to save the results
#     Outputs:
#     None
#     """
#     path_to_save = path_to_save+'unsupervised/'
#     if(not (os.path.isdir(path_to_save))):
#         print(path_to_save, "  was not present, creating the folder...")
#         os.makedirs(path_to_save)

#     # Extract only numbers from the name
#     model = HomographyModel()
#     CheckPoint = torch.load(ModelPath)
#     model.load_state_dict(CheckPoint["model_state_dict"])
#     print("Loaded latest checkpoint....")

#     all_labels = pd.read_csv(BasePath+ 'Test_generated/' + "TestH4.csv", index_col =False)
#     all_labels = all_labels.to_numpy()
#     all_patchNames = pd.read_csv(BasePath+ 'Test_generated/' + "TestImageFileNames.csv")
#     all_patchNames = all_patchNames.to_numpy()

#     X_test = []
#     for p in all_patchNames:
#     #     print(p)
#         tPatchA = cv2.imread(BasePath+'Test_generated/'+"patchA/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
#         tPatchA = torch.from_numpy((tPatchA).permute(2, 0, 1).to(torch.float32)) /255
#         tPatchB = cv2.imread(BasePath+'Test_generated/'+"patchB/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
#         tPatchB = torch.from_numpy((tPatchB).permute(2, 0, 1).to(torch.float32)) /255
#         tPatch = np.dstack((tPatchA, tPatchB))    
#         X_test.append(tPatch)

#     X_test = np.array(X_test)   
#     Y_true = all_labels

#     print("Shape of X_test and Y_test ", X_test.shape,Y_true.shape)

#     Y_true = torch.from_numpy(all_labels).float()  # or torch.tensor(all_labels).float()

#     X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).to(torch.float32)
#     with torch.no_grad():
#         Y_pred = model(X_test)
    
#     np.save(path_to_save+"Y_Pred.npy",Y_pred)
#     Y_true = Y_true.repeat(4, 1)
#     mae = nn.functional.l1_loss(Y_pred, Y_true).item()  # Mean Absolute Error
#     mse = nn.functional.mse_loss(Y_pred, Y_true).item()  # Mean Squared Error
#     rmse = torch.sqrt(mse).item()  # 

#     print("Mean Absolute Error: ", mae)
#     print("Mean Squared Error: ", mse)
#     print("Root Mean Squared Error: ", rmse)
    
#     return None

def supervised_test(BasePath, ModelPath, path_to_save):
    
    # Add the missing import statement for the `model` class here

    path_to_save = path_to_save+'supervised/'
    if(not (os.path.isdir(path_to_save))):
        print(path_to_save, "  was not present, creating the folder...")
        os.makedirs(path_to_save)

    # Extract only numbers from the name
    model = HomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print("Loaded latest checkpoint....")

    all_labels = pd.read_csv(BasePath+ 'Test_generated/' + "TestH4.csv", index_col =False)
    all_labels = all_labels.to_numpy()
    all_patchNames = pd.read_csv(BasePath+ 'Test_generated/' + "TestImageFileNames.csv")
    all_patchNames = all_patchNames.to_numpy()

    X_test = []
    for p in all_patchNames:
        tPatchA = cv2.imread(BasePath+'Test_generated/'+"patchA/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
        tPatchA = torch.from_numpy((tPatchA).permute(2, 0, 1).to(torch.float32)) /255
        tPatchB = cv2.imread(BasePath+'Test_generated/'+"patchB/"+ str(p[0]), cv2.IMREAD_GRAYSCALE)
        tPatchB = torch.from_numpy((tPatchB).permute(2, 0, 1).to(torch.float32)) /255
        tPatch = np.dstack((tPatchA, tPatchB))    
        X_test.append(tPatch)

    X_test = np.array(X_test)   
    Y_true = all_labels

    print("Shape of X_test and Y_test ", X_test.shape,Y_true.shape)

    Y_true = torch.from_numpy(all_labels).float()  # or torch.tensor(all_labels).float()

    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).to(torch.float32)
    with torch.no_grad():
        Y_pred = model(X_test)
    
    np.save(path_to_save+"Y_Pred.npy",Y_pred)
    Y_true = Y_true.repeat(4, 1)
    mae = nn.functional.l1_loss(Y_pred, Y_true).item()  # Mean Absolute Error
    mse = nn.functional.mse_loss(Y_pred, Y_true).item()  # Mean Squared Error
    rmse = torch.sqrt(mse).item()  # 

    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    
    return None


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints/9model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../Data/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/DirNamesTest.txt",
        help="Path of labels file, Default:./TxtFiles/DirNamesTest.txt",
    )
    Parser.add_argument('--SavePath', dest='SavePath', default='../Data/TestResults/', help='Path of labels file, Default: ./Results/')
    Parser.add_argument('--ModelType', default='supervised', help='supervised or unsupervised, Default:supervised')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    SavePath = Args.SavePath
    ModelType = Args.ModelType

    if ModelType == 'unsupervised':
        print('Unsupervised Model')
        unsupervised_test(BasePath, ModelPath, SavePath)
        print('Results/unsupervised folder..')

    else:
        print('Supervised Model')
        supervised_test(BasePath, ModelPath, SavePath)

        dir = pd.read_csv(BasePath+'Test_generated/'+"TestH4.csv", index_col =False) 
        rand_i = np.random.randint(0,len(dir)-1, size=5)
        # for i in rand_i:
        #     comparison = view_sup(i, BasePath, SavePath)
        #     cv2.imwrite(SavePath+'supervised/comparison'+ str(i)+'.png',comparison)

        print('Results/supervised folder..')
    
if __name__ == '__main__':
    main()
