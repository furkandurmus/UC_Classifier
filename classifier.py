#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 00:52:01 2021

@author: capsule2232
"""
import torch
from torchvision import datasets, transforms
from torchvision.ops import focal_loss, sigmoid_focal_loss
from torch.utils import data
import torch.nn as nn
from torch.nn import functional as F
import test_loader

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiplicativeLR
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import timm
from timm.data.auto_augment import rand_augment_transform
from AutoAugment.autoaugment import ImageNetPolicy
from RandAugment import RandAugment
import ttach as tta

import focal_loss  
import argparse
import copy
from tqdm import tqdm
import time
import os

import glob
import time
import os
import copy
import sys
import pandas as pd
import numpy as np

import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
from multiprocessing import Process, freeze_support



parser = argparse.ArgumentParser()


parser.add_argument("--train_data", default="/home/capsule2232/classification/mayo_final/train")
parser.add_argument("--test_data", default="/home/capsule2232/classification/mayo_final/test")
parser.add_argument("--out_dir", 
                default="/home/capsule2232/classification/ulcer_real_train",
                help="Main output dierectory")

parser.add_argument("--action", type=str, choices=["train", "test"], default="test" )
parser.add_argument("--augmentation_size", type=tuple, default=(224,224), help='Resize the augmented image transform')
parser.add_argument("--bs", type=int, default=16, help="Batch Size")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epoch_number", type=int, default=100)
parser.add_argument("--class_number", type=int, default=4)
opt = parser.parse_args()



print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#create subfolder due time
timestr = time.strftime("%Y%m%d-%H%M%S")
tensorboard_output = os.path.join(opt.out_dir, timestr)
os.makedirs(timestr)

writer = SummaryWriter(timestr)


#list of the classifiers
model_list = timm.list_models()



def prepare_data():
    
    data_transforms = {
    'standart': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    
    'test': transforms.Compose([
        transforms.Resize(opt.augmentation_size),
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    
    'imagenet': transforms.Compose([
        transforms.Resize(opt.augmentation_size),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
            )])
    

    }
    
    data_transforms["standart"].transforms.insert(0, RandAugment(2, 9))
    train_set = datasets.ImageFolder(root=opt.train_data, transform=data_transforms["standart"])
    test_set = datasets.ImageFolder(root=opt.test_data, transform=data_transforms["test"])
    
    train_loader = data.DataLoader(train_set, batch_size=opt.bs, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=opt.bs, shuffle=False)
    
    print("Train dataset size = ", len(train_set))
    print("Test dataset size = ", len(test_set))
    
    return {"train":train_loader, "test":test_loader, "dataset_size":{"train":len(train_set), "test":(len(test_set))}}





def train_model(model, optimizer, criterion, dataloaders, scheduler, best_acc=0.0, start_epoch=0):
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in tqdm(range(start_epoch, start_epoch + opt.epoch_number)):
        
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                dataloader = dataloaders["train"]
            
            else:
                model.eval()
                dataloader = dataloaders["test"] 
                
            
            
            running_loss = 0.0
            running_corrects = 0
        
            for i, data_ in tqdm(enumerate(dataloader, 0)):
            
                inputs, labels = data_
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                optimizer.zero_grad()
            
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # breakpoint()

                    # out = outputs
                    # outputs = outputs[1]
                   # breakpoint()
                    # breakpoint()
                    # try:
                    #     _, preds = torch.max(outputs, 1)
                    #     loss = criterion(outputs, labels)

                    # except IndexError:
                    #     _, preds = torch.max(outputs, 0)
                    #     breakpoint()
                    #     loss = criterion(out, labels)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataloaders["dataset_size"][phase]
            epoch_acc = running_corrects.double() / dataloaders["dataset_size"][phase]

            # update tensorboard writer
            writer.add_scalars("Loss", {phase:epoch_loss}, epoch)
            writer.add_scalars("Accuracy" , {phase:epoch_acc}, epoch)

             # update the lr based on the epoch loss
            if phase == "test": 

                # keep best model weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    best_epoch_acc = epoch_acc
                    print("Found a better model")

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LR", lr, epoch)
                scheduler.step(epoch_loss) 
            

            print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f | Acc:\t %.4f '
                      % (epoch, phase, epoch_loss, epoch_acc))
    
    save_model(best_model_wts, best_epoch, best_epoch_loss, best_epoch_acc)            


def prepare_model():
    
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=opt.class_number)
    # num_ftrs = model.head_dist.l.in_features
    # output_features = opt.class_number
    # model.head.l.out_features = output_features
    # model.head_dist.l.out_features = output_features
    model = model.to(device)
    
    
    return model


def run_train(retrain=False):
    model = prepare_model()
    
    dataloaders = prepare_data()

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr , weight_decay=opt.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr )

    # criterion =  nn.MSELoss() # backprop loss calculation
    # criterion = nn.CrossEntropyLoss() # weight=weights
    criterion = focal_loss.FocalLoss()
    # criterion = sigmoid_focal_loss()
    # criterion_validation = nn.L1Loss() # Absolute error for real loss calculations

    # LR shceduler
    #scheduler = ReduceLROnPlateau(optimizer, mode="min", verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = MultiplicativeLR()
    # call main train loop

    if retrain:
        # train from a checkpoint
        checkpoint_path = input("Please enter the checkpoint path:")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        acc = checkpoint["acc"]
        train_model(model,optimizer,criterion, dataloaders, scheduler, best_acc=acc, start_epoch=start_epoch)

    else:
        train_model(model,optimizer,criterion, dataloaders, scheduler, best_acc=0.0, start_epoch=0)





def save_model(model_weights,  best_epoch,  best_epoch_loss, best_epoch_acc):
   
    check_point_name = timestr + "_epoch:{}.pt".format(best_epoch) # get code file name and make a name
    check_point_path = os.path.join(timestr, check_point_name)
    # save torch model
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model_weights,
        # "optimizer_state_dict": optimizer.state_dict(),
        # "train_loss": train_loss,
        "loss": best_epoch_loss,
        "acc": best_epoch_acc,
    }, check_point_path)




def test_model():
    
    test_model_checkpoint = input("Please enter the path of test model:")
    checkpoint = torch.load(test_model_checkpoint)

    model = prepare_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    


    dataloaders = prepare_data()
    test_dataloader = dataloaders["test"]

    # TO collect data
    correct = 0
    total = 0
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)



    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):

            inputs, labels = data



            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_labels_d = torch.cat((all_labels_d, labels), 0)
            all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
            all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)


    print('copying some data back to cpu for generating confusion matrix...')
    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()  # to('cpu')
    testset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')


    #return y_predicted, testset_predicted_probabilites, all_timePerFrame_host


    cm = confusion_matrix(y_true, y_predicted)  # confusion matrix



    print('Accuracy of the network on the %d test images: %f %%' % (total, (
            100.0 * correct / total)))

    print(cm)

    print("taking class names to plot CM")

    class_names = test_dataloader.dataset.classes #test_datasets.classes  # taking class names for plotting confusion matrix

    

    ##################################################################
    # classification report
    #################################################################
    print(classification_report(y_true, y_predicted, target_names=class_names))

    ##################################################################
    # Standard metrics for medico Task
    #################################################################
    print("Printing standard metric for medico task")

    print("Accuracy =",mtc.accuracy_score(y_true, y_predicted))
    print("Precision score =", mtc.precision_score(y_true,y_predicted, average="weighted"))
    print("Recall score =", mtc.recall_score(y_true, y_predicted, average="weighted"))
    print("F1 score =", mtc.f1_score(y_true, y_predicted, average="weighted"))
    print("Specificity =")
    print("MCC =", mtc.matthews_corrcoef(y_true, y_predicted))

    ##################################################################
    # Standard metrics for medico Task
    #################################################################
    print("Printing standard metric for medico task")

    

    print("1. Recall score (REC) =", mtc.recall_score(y_true, y_predicted, average="weighted"))
    print("2. Precision score (PREC) =",
            mtc.precision_score(y_true, y_predicted, average="weighted"))
    print("3. Specificity (SPEC) =")
    # print("4. Accuracy (ACC) =", mtc.accuracy_score(y_true, y_predicted, weights))
    print("5. Matthews correlation coefficient(MCC) =", mtc.matthews_corrcoef(y_true, y_predicted))

    print("6. F1 score (F1) =", mtc.f1_score(y_true, y_predicted, average="weighted"))

    
    print('Finished.. ')



if __name__ == '__main__':
    print("Data Preperation...")
    data_loaders = prepare_data()
    print("Data is ready")
    
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
       # pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_train(retrain=True)
    elif opt.action == "test":
        test_model()
       # pass
    # elif opt.action == "test":
    #     print("Inference process is strted..!")
    #     prepare_model()
    #     test_model()


    # Finish tensorboard writer
    writer.close()