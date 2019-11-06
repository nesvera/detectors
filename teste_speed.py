import torch
from models import SSD_MobileNet
from utils import utils

from torchsummary import summary

import time

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # build detector
    base_pretrained = None
    config_num_classes = 21
    model = SSD_MobileNet.SSDMobileNet(base_pretrained, config_num_classes)
    
    model = model.to(device)
    model.eval()

    summary(model, (3, 224, 224))
    
    print("Press ENTER to continue")
    input()

    prediction_time = utils.Average()    
    start_time = 0

    with torch.no_grad():
        for i in range(1000):

            image = torch.randn(1, 3, 224, 224).to(device)

            start_time = time.time()

            prediction = model(image)

            prediction_time.add_value(time.time()-start_time)

            print("[{0}/{1}]".format(i,1000))

    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))
    
    
'''
Mobilenet DW alpha=1.0
Total params: 3,237,726
Average prediction time: 0.0030
Min prediction time: 0.0029
Max prediction time: 0.0073

Mobilenet Conv2d alpha=1.0
Total params: 28,299,838
Average prediction time: 0.0019
Min prediction time: 0.0019
Max prediction time: 0.0052

Mobilenet Conv2d alpha=0.5
Total params: 7,085,870
Average prediction time: 0.0019
Min prediction time: 0.0019
Max prediction time: 0.0050

'''
