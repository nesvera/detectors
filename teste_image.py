import torch
import torchvision.transforms.functional as FT

from models import SSD_MobileNet

import argparse
import numpy as np
import cv2
import os
from torchsummary import summary
from PIL import Image

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                       dest="model",
                       help="Model path",
                       required=True)
    parser.add_argument("--image",
                       dest="image",
                       help="Image path",
                       required=True)
    
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image

    if os.path.exists(model_path) == False:
        print("Error: Model file was not found!")
        exit(1)

    if os.path.exists(image_path) == False:
        print("Error: Image file was not found!")
        exit(1)

    # load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint["model"]

    #summary(model, (3, 224, 224))
    
    # load image
    img = cv2.imread(image_path)
    pil_image = Image.fromarray(img)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    image = FT.normalize(FT.to_tensor(FT.resize(pil_image, (224,224))), mean=mean, std=std)
    image = image.to(device).unsqueeze(0)   # [1, 3, 224, 224] 

    # Forward prop
    pred_locs, pred_scores = model(image) 

    det_boxes, det_labels, det_scores = model.detect_objects(pred_locs, pred_scores,
                                                             min_score=0.99,
                                                             max_overlap=0.1,
                                                             top_k=10)

    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor([pil_image.width, pil_image.height, pil_image.width, pil_image.height]).unsqueeze(0)
    det_boxes = det_boxes*original_dims
    
    print("Prediction: ")
    print(det_labels)
    print(det_scores)

    for i in range(det_boxes.size(0)):
        bb = det_boxes[i]        
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,0), 2)
        
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
