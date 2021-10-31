#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 14:26:41 2021

@author: capsule2232
"""
import torch
import albumentations
import numpy as np
from PIL import Image

class test_Dataset(torch.utils.data.Dataset):
    def __init__(self, ids, image_ids):
        self.ids = ids
        self.image_ids = image_ids # list of testset image ids
        #test data augmentations
        self.aug = albumentations.Compose([
                    albumentations.RandomResizedCrop(256, 256),
                    albumentations.Transpose(p=0.5),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.VerticalFlip(p=0.5),
                    albumentations.HueSaturationValue(
                        hue_shift_limit=0.2, 
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2, 
                        p=0.5
                    ),
                    albumentations.RandomBrightnessContrast(
                        brightness_limit=(-0.1,0.1), 
                        contrast_limit=(-0.1, 0.1), 
                        p=0.5
                    ),
                    albumentations.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    )
                ], p=1.)
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        # converting jpg format of images to numpy array
        img = np.array(Image.open('_PATH_' + self.image_ids[index]))
        #Applying augmentations to numpy array
        img = self.aug(image = img)['image']
        # converting to pytorch image format & 2,0,1 because pytorch excepts image channel first then dimension of image
        img = np.transpose(img , (2,0,1)).astype(np.float32) 
        
        # finally returning image tensor and its image id
        return torch.tensor(img, dtype = torch.float) , self.image_ids[index]
    
    
    

if __name__=="__main__":

    root_path = "/home/capsule2232/classification/mayo_final/train"

    test = test_Dataset(root_path)
    print(test[0][2].split("/"))