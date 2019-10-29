from utils import utils, datasets
import torch
import torchvision.transforms.functional as FT

from torchsummary import summary

from PIL import Image
import cv2

from models import SSD_MobileNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect(model, in_image, min_score, max_overlap, top_k):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    image = FT.normalize(FT.to_tensor(FT.resize(in_image, (224,224))), mean=mean, std=std)
    image = image.to(device).unsqueeze(0)   # [1, 3, 224, 224]

    # Forward prop
    pred_locs, pred_scores = model(image) 

    det_boxes, det_labels, det_scores = model.detect_objects(pred_locs, pred_scores,
                                                             min_score=min_score,
                                                             max_overlap=max_overlap,
                                                             top_k=top_k)

    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor([in_image.width, in_image.height, in_image.width, in_image.height]).unsqueeze(0)
    det_boxes = det_boxes*original_dims

    return det_boxes, 0, 0

if __name__ == "__main__":

    checkpoint = torch.load("/home/feaf-seat-1/Documents/nesvera/detectors/experiments/detector_mobilenet_dw_224/BEST_detector_mobilenet_dw_224.pth.tar")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    summary(model, (3,224,224))
    
    # receive image from camera
    cap = cv2.VideoCapture("/home/feaf-seat-1/Downloads/carros.jpeg")

    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        # convert (numpy bgr -> numpy rgb -> PIL image)
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)

        bounding_boxes, labels, scores = detect(model, pil_image, min_score=0.15, max_overlap=0.5, top_k=200)

        for bb in bounding_boxes:
            bb = bb.detach().numpy().astype('int')
            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0,0,255), 2)

            print(bb)

        print("caaaaa")
        cv2.imshow("image", frame)
        cv2.waitKey(0)
        
            


