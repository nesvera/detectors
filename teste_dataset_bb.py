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
    image_width = 224
    image_height = 224

    base_pretrained = None
    num_classes = 21

    # build detector
    model = SSD_MobileNet.SSDMobileNet(base_pretrained, num_classes)

    # get priors of the model
    model = model.to(device)
    priors_boxes_cxcy = model.create_prior()
    priors_boxes_xy = utils.cxcy_to_xy(priors_boxes_cxcy)

    # ------------------------
    #       Dataloaders
    # ------------------------  
    #data_folder = "/home/feaf-seat-1/Documents/nesvera/object_detection/a-PyTorch-Tutorial-to-Object-Detection"
    data_folder = "/home/nesvera/Documents/neural_nets/object_detection/a-PyTorch-Tutorial-to-Object-Detection"
    train_dataset = datasets.PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=2,
                                               pin_memory=True)

    for i in range(len(train_dataset)):

        print("priors: ", priors_boxes_cxcy.size())

        (images, boxes, labels, _) = train_dataset[i]

        image = images.permute(1,2,0).numpy()

        boxes_cx_cy = utils.xy_to_cxcy(boxes)
        boxes_xy = utils.cxcy_to_xy(boxes_cx_cy)

        true_locs = torch.zeros((1, boxes_xy.size(0), 4),
                                dtype=torch.float).to(device)                   # [N, n_priors, 4]
        true_classes = torch.zeros((1, boxes_xy.size(0)),
                                dtype=torch.long).to(device)                    # [N, n_priors]

        n_objects = boxes.size(0)

        overlap = utils.find_jaccard_overlap(boxes, priors_boxes_xy)      # [n_objects, n_prios]

        # create a vector with an object that has the max overlap for each prior
        overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # [n_priros]

        # Problems:
        # 1. Suppose that there are some objects near each other. It is possible that one of
        # objects does not have any good overlap for anyone of the priors. Them it will not
        # apper in any of the object_for_each_prior vector

        # First, find the prior that has the maximum overlap for each object
        _, prior_for_each_object = overlap.max(dim=1)                       # [n_objects]

        # Then, assign each object to the corresponding maximum overlap prior
        object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

        # Then, assign maximum overlap for these objects
        overlap_for_each_prior[prior_for_each_object] = 1.

        # Encode coordinates from xmin,ymin,xmax,ymax to center-offset
        true_locs = utils.cxcy_to_gcxgcy(utils.xy_to_cxcy(boxes[object_for_each_prior]),
                                            priors_boxes_cxcy)               # [n_priors, 4]
        true_locs = true_locs.unsqueeze(0)

        pred_cls = torch.ones((priors_boxes_xy.size(0), 21),
                              dtype=torch.float).to(device)
        pred_cls = pred_cls.unsqueeze(0)

        det_boxes, det_labels, det_scores = model.detect_objects(true_locs, pred_cls,
                                                             min_score=0,
                                                             max_overlap=0.5,
                                                             top_k=200)

        # bounding box
        for j in range(boxes.shape[0]):
            p0 = (int(boxes[j, 0]*image_width), int(boxes[j, 1]*image_height))
            p1 = (int(boxes[j, 2]*image_width), int(boxes[j, 3]*image_height))

            image = cv2.rectangle(image, p0, p1, (255,0,0), 3)

        image = image.get()

        decoded_locs = det_boxes[0]

        k = 0
        #for k in range(decoded_locs.size(0)):
        while True:
            k += 100
            
            p0 = (int(decoded_locs[k, 0]*image_width), int(decoded_locs[k, 1]*image_height))
            p1 = (int(decoded_locs[k, 2]*image_width), int(decoded_locs[k, 3]*image_height))

            img = cv2.rectangle(image.copy(), p0, p1, (0,255,0), 1)

            cv2.imshow("image", img)
            cv2.waitKey(0)

        '''
        """ Printar os priors """

        overlap = utils.find_jaccard_overlap(boxes, priors_boxes_xy)      # [n_objects, n_prios]

        # create a vector with an object that has the max overlap for each prior
        overlap_for_each_object, prior_for_each_object = overlap.max(dim=1)  # [n_priros]

        print("Number of objects: ", boxes.size())
        print("Overlap_for_each_object: ", overlap_for_each_object.size(), prior_for_each_object.size())
        print("Overlap: ", overlap_for_each_object)

        boxes = boxes.numpy()
        boxes_xy = boxes_xy.numpy()

        # bounding box
        for j in range(boxes.shape[0]):
            p0 = (int(boxes[j, 0]*image_width), int(boxes[j, 1]*image_height))
            p1 = (int(boxes[j, 2]*image_width), int(boxes[j, 3]*image_height))

            image = cv2.rectangle(image, p0, p1, (255,0,0), 3)

        image = image.get()

    
        priors_boxes = utils.cxcy_to_xy(priors_boxes_cxcy)

        conv_d56 = 56*56*1
        conv_d28 = 28*28*3
        conv_d14 = 14*14*5
        conv_d7 = 7*7*5
        conv_d5 = 5*5*5
        conv_d3 = 3*3*3
        conv_d1 = 1*1*3

        priors_boxes_d56 = priors_boxes[0:conv_d56]
        priors_boxes_d28 = priors_boxes[conv_d56:(conv_d56+conv_d28)]
        priors_boxes_d14 = priors_boxes[(conv_d56+conv_d28):(conv_d56+conv_d28+conv_d14)]
        priors_boxes_d7 =  priors_boxes[(conv_d56+conv_d28+conv_d14):(conv_d56+conv_d28+conv_d14+conv_d7)]
        priors_boxes_d5 =  priors_boxes[(conv_d56+conv_d28+conv_d14+conv_d7):(conv_d56+conv_d28+conv_d14+conv_d7+conv_d5)]
        priors_boxes_d3 =  priors_boxes[(conv_d56+conv_d28+conv_d14+conv_d7+conv_d5):(conv_d56+conv_d28+conv_d14+conv_d7+conv_d5+conv_d3)]

        priors_boxes = priors_boxes_d56

        for k in range(priors_boxes.size(0)):
            obj = overlap[:, k]
            print(obj)
            
            p0 = (int(priors_boxes[k, 0]*image_width), int(priors_boxes[k, 1]*image_height))
            p1 = (int(priors_boxes[k, 2]*image_width), int(priors_boxes[k, 3]*image_height))

            img = cv2.rectangle(image.copy(), p0, p1, (0,255,0), 1)

            cv2.imshow("image", img)
            cv2.waitKey(0)


        # prior
        for j in range(boxes.shape[0]):
            prior_n = prior_for_each_object[j]
            box = priors_boxes_cxcy[prior_n].unsqueeze(0)
            box = utils.cxcy_to_xy(box).squeeze(0)
            box = box.numpy()

            p0 = (int(box[0]*image_width), int(box[1]*image_height))
            p1 = (int(box[2]*image_width), int(box[3]*image_height))

            image = cv2.rectangle(image, p0, p1, (0,0,255), 1)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        '''

if __name__ == "__main__":
    main()
