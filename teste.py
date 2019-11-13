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
    
    #image = FT.normalize(FT.to_tensor(FT.resize(in_image, (224,224))), mean=mean, std=std)
    #image = image.to(device).unsqueeze(0)   # [1, 3, 224, 224]
    image = in_image

    # Forward prop
    pred_locs, pred_scores = model(image) 

    det_boxes, det_labels, det_scores = model.detect_objects(pred_locs, pred_scores,
                                                             min_score=min_score,
                                                             max_overlap=max_overlap,
                                                             top_k=top_k)

    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor([224, 224, 224, 224]).unsqueeze(0)
    det_boxes = det_boxes*original_dims

    return det_boxes, det_labels, det_scores

if __name__ == "__main__":

    checkpoint = torch.load("/home/feaf-seat-1/Documents/nesvera/detectors/experiments/detector_mobilenet_dw_224/BEST_detector_mobilenet_dw_224.pth.tar")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    summary(model, (3,224,224))
    
    # receive image from camera
    #cap = cv2.VideoCapture("/home/feaf-seat-1/Downloads/carros.jpeg")

    data_folder = "/home/feaf-seat-1/Documents/nesvera/object_detection/a-PyTorch-Tutorial-to-Object-Detection"
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

        (images, boxes, labels, _) = train_dataset[i]

        images = images.unsqueeze(0).to(device)

        pred_bb, pred_labels, pred_scores = detect(model, images, min_score=0.7, max_overlap=0.2, top_k=200)
        
        #print(pred_scores)
        print("True labels: ", labels)
        print("Pred labels: ", pred_labels)
        print()

        image = images.to('cpu').squeeze(0).permute(1,2,0).numpy()

        for bb in pred_bb:
            bb = bb.detach().numpy().astype('int')
            image = cv2.rectangle(image.copy(), (bb[0], bb[1]), (bb[2], bb[3]), (0,0,255), 2)

            
            cv2.imshow("image", image)
            cv2.waitKey(0)

            


