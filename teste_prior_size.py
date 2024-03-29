import torch
import torch.nn as nn
from torchsummary import summary

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

    # get priors of the model
    model = model.to(device)
    priors_boxes_cxcy = model.create_prior()

    # create a prediction tensor
    # duo to the hardtanh activation function, the predictions 
    # are limited between [-1,1]
    #prediction = torch.zeros(priors_boxes_cxcy.size())        # all values 0
    prediction = torch.ones(priors_boxes_cxcy.size())        # all values 1
    #prediction = torch.ones(priors_boxes_cxcy.size())*(-1)   # all values -1

    decoded_pred = utils.gcxgcy_to_cxcy(prediction.to(device), priors_boxes_cxcy)
    decoded_pred = utils.cxcy_to_xy(decoded_pred)

    print("Priors tensor: ", priors_boxes_cxcy.size())
    print("Prediction tensor: ", prediction.size())
    print("Decoded tensor: ", decoded_pred.size())

    image = cv2.imread("/home/nesvera/Documents/neural_nets/object_detection/detectors/images/cut_1.png")
    
    for i in range(priors_boxes_cxcy.size(0)):
        p0 = (int(decoded_pred[i][1]), int(decoded_pred[i][0]))
        p1 = (int(decoded_pred[i][3]), int(decoded_pred[i][2]))

        print(p0, p1)

        img = cv2.rectangle(image.copy(), p0, p1, (255,0,0), 3)

        cv2.imshow("image", img)
        cv2.waitKey(0)
    
if __name__ == "__main__":
    main()
