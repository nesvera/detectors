import torch
import torch.nn as nn
from torchsummary import summary

from classifiers.models import MobileNet
from models import SSD_MobileNet
from utils import datasets, loss_function, utils

import argparse
import numpy as np
import os
import yaml
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # ------------------------
    #    Build/Load model
    # ------------------------

    base_pretrained = None
    num_classes = 21

    # build detector
    model = SSD_MobileNet.SSDMobileNet(base_pretrained, num_classes)

    model = model.to(device)
    priors_boxes = model.create_prior()

    prior_b = priors_boxes
    print("prior", prior_b.size())

    feat_map_d56 = 56*56*2*1

    for i in range(prior_b.size(0)):
        print(prior_b[i])
        input()
    
if __name__ == "__main__":
    main()
