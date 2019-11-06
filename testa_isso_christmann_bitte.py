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

    model_trt = torch2trt(model, [x])           # convert model to tensorRT
    
    prediction_time = utils.Average()    
    start_time = 0

    with torch.no_grad():
        for i in range(1000):

            image = torch.randn(1, 3, 224, 224).cuda()

            start_time = time.time()

            prediction_model = model(image)

            prediction_time.add_value(time.time()-start_time)

    print("Model running with pytorch API")
    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))


    with torch.no_grad():
        for i in range(1000):

            image = torch.randn(1, 3, 224, 224).cuda()

            start_time = time.time()

            prediction_trt = model_trt(image)

            prediction_time.add_value(time.time()-start_time)

    print("Model running with TensorRT")
    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))