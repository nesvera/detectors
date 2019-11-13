import torch
from models import SSD_MobileNet, SSD_VGG_16, SSD_MobileNet_DW
from utils import utils

from torchsummary import summary

import time

if __name__ == "__main__":

    model = "mobilenet"             # ssd based in the mobilenet with standard convolutional layers
    #model = "mobilenet_dw"         # ssd based in the mobilenet with depthwise convolutional layers
    #model = "vgg"                  # ssd based in the vgg16

    alpha = 0.75

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model == "mobilenet":
        base_pretrained = None
        config_num_classes = 21
        model = SSD_MobileNet.SSDMobileNet(base_pretrained, config_num_classes, alpha=alpha)
        in_dim = (3, 224, 224)

    elif model == "mobilenet_dw":
        base_pretrained = None
        config_num_classes = 21
        model = SSD_MobileNet_DW.SSDMobileNet(base_pretrained, config_num_classes, alpha=alpha)
        in_dim = (3, 224, 224)
    
    elif model == "vgg":
        config_num_classes = 21
        model = model = SSD_VGG_16.SSD300(n_classes=config_num_classes)
        in_dim = (3, 300, 300)

    model = model.to(device)
    model.eval()

    summary(model, in_dim)
    
    print("Press ENTER to continue")
    input()

    prediction_time = utils.Average()    
    start_time = 0

    with torch.no_grad():
        for i in range(1000):

            # generate a random image, predict over it, and count the time
            image = torch.randn(1, in_dim[0], in_dim[1], in_dim[2]).to(device)

            start_time = time.time()

            prediction = model(image)
            print()

            pred_period = time.time()-start_time
            prediction_time.add_value(pred_period)

            print("[{0}/{1}]".format(i,1000))
            print(pred_period)

    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))
    print("Sum prediction time: {0:.4f}".format(prediction_time.get_sum()))    
    
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
