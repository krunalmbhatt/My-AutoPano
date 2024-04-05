#!/usr/bin/evn python


# Code starts here:

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
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

def getPAPBH4(img, patch_size=128, max_perturbation=32, limit=42):
    """
    Generate a pair of patches from the input image.
    :param img: input image (HxWxC)
    :param patch_size: size of the patch (Default: 128)
    :param max_perturbation: maximum perturbation allowed (shift allowed to obtain patchB)
    :param border_limit: border limit for choosing where to crop from.
    :return: Patch_a, Patch_b, H4, imageB, points
    """

    h, w = img.shape[:2]                                                                                    # Get the height and width of the image
    minimum_size = 2*limit + patch_size + 1                                                                 # Minimum size for the image

    if w > minimum_size and h > minimum_size:                                           
        max = patch_size + limit                                                               # Maximum size of the image                           
        x = np.random.randint(limit, w - max)                                                  # Randomly generate a number between border_limit and w-max
        y = np.random.randint(limit, h - max)                                                  # Randomly generate a number between border_limit and h-max                                                
        
        points = np.array([[x, y], [x, patch_size + y], [patch_size + x, y], [patch_size + x, patch_size + y]])    # Coordinates of the four corners of the patch A
        point1 = np.zeros_like(points)                                                                           # Initialize point1 with zeros

        # Randomly perturb the points to obtain patch B
        for i, pt in enumerate(points):
            point1[i][0] = pt[0] + np.random.randint(-max_perturbation, max_perturbation)
            point1[i][1] = pt[1] + np.random.randint(-max_perturbation, max_perturbation)

        # Wrap the image using the perturbed points
        H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(points), np.float32(point1)))
        
        imageB = cv2.warpPerspective(img, H, (w, h))

        
        patch_A = img[y:y + patch_size, x:x + patch_size]
        patch_B = imageB[y:y + patch_size, x:x + patch_size]
        H4 = (point1 - points).astype(np.float32)

        return patch_A, patch_B, H4, imageB, np.dstack((points, point1))
    else:
        return None, None, None, None, None



def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Generates data for supervised learning network and unsupervised learning network
    Path to the data directory

    """
    # base_path = './Data/Train/'
    base_path = './Data/Train/'
    patch_size = 128
    max_perturbation = 32
    limit = 42
    image_set = ['Train','Val', 'Test']

    for imgset in image_set:
        count=0

        if imgset == 'Train':
            print("Training data...")
            print("Generating Train data ......")
            path = './Data/Train/'
            generatedimgs = './Data/Train_generated/'
            loopCount = 5001 
            
        elif imgset == 'Val':
            print("Generating Validation data ......")
            path = './Data/Val/'
            generatedimgs = './Data/Val_generated/'
            loopCount = 1000 

        else:
            print("Generating Test data ......")
            path = './Data/Test/Phase2/'    # the released test folder with 1000 images was named as Phase2
            generatedimgs = './Data/Test_generated/'
            loopCount = 1001 #Val folder had 1000 images
            
        if(not (os.path.isdir(generatedimgs))):
            print(generatedimgs, "  was not present, creating the folder...")
            os.makedirs(generatedimgs)
        
        H4_list = []
        image_name_list = [] 
        pointsList = []
        print("Begin Data Generation .... ")
        
        for i in range(1,loopCount):

            #random_ind = np.random.choice(range(1, 5000), replace= False)
            imageA = cv2.imread(path + str(i) + '.jpg')
            resizedA = cv2.resize(imageA, (320,240), interpolation = cv2.INTER_AREA)

            patch_A, patch_B, H4, _, points = getPAPBH4(resizedA, patch_size,max_perturbation, limit) 
                
            if ((patch_A is None)&(patch_B is None)&(H4 is None)):
                print("Error.. Image ignored: ", i)
                count+=1
            else:
                if(not (os.path.isdir(generatedimgs +'patchA/'))):
                    print(" Subdirectories inside  ", base_path, " were not present.. creating the folders...")
                    os.makedirs(generatedimgs +'patchA/')
                    os.makedirs(generatedimgs +'patchB/')
                    os.makedirs(generatedimgs +'resizedA/')
 
                pathA = generatedimgs +'patchA/' + str(i) + '.jpg'
                pathB = generatedimgs +'patchB/' + str(i) + '.jpg'
                image_path_A = generatedimgs +'resizedA/' + str(i) + '.jpg'

                cv2.imwrite(pathA, patch_A)
                cv2.imwrite(pathB, patch_B)
                cv2.imwrite(image_path_A, imageA)

                H4_list.append(np.hstack((H4[:,0] , H4[:,1])))
                pointsList.append(points)
                image_name_list.append(str(i) + '.jpg')

                
            files_path = generatedimgs
            df = pd.DataFrame(H4_list)
            df.to_csv(files_path+ imgset+ "H4.csv", index=False)
            np.save(files_path+ imgset+ "H4.npy", np.array(H4_list))
            np.save(files_path+ imgset+ "pointsList.npy", np.array(pointsList))
                
            df = pd.DataFrame(image_name_list)
            df.to_csv(files_path+ imgset+ "ImageFileNames.csv", index=False)

        print("Saved")
        print("done")
        print("No. of labels: ", len(H4_list),"No. of images: ", len(image_name_list), "No. of points: ", np.array(pointsList).shape,  "No. of patches: ",(i-count))

if __name__ == "__main__":
    main()
