import torch
from torch2trt import torch2trt

from models import SSD_MobileNet
from utils import utils
import time

if __name__ == "__main__":
    base_pretrained = None
    config_num_classes = 21
    model = SSD_MobileNet.SSDMobileNet(base_pretrained, config_num_classes).eval().cuda()



    x = torch.ones((1,3,224,224)).cuda()

    model_trt = torch2trt(model, [x])
    
    prediction_time = utils.Average()    
    start_time = 0

    with torch.no_grad():
        for i in range(1000):

            image = torch.randn(1, 3, 224, 224).cuda()

            start_time = time.time()

            #prediction_model = model(image)
            prediction_trt = model_trt(image)

            # check the output against PyTorch
            #print(torch.max(torch.abs(prediction_model[0] - prediction_trt[0])))
            #input()

            prediction_time.add_value(time.time()-start_time)

            #print("[{0}/{1}]".format(i,1000))
            
    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))

