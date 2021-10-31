#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:55:12 2021

@author: capsule2232
"""

from vit_pytorch import ViT
import torch
from dataset.Dataloader_with_path import ImageFolderWithPaths as dataset
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default="./ulcer_data")


v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
        )

