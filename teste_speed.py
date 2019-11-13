import torch
from models import SSD_MobileNet, SSD_VGG_16, SSD_MobileNet_DW
from utils import utils

from torchsummary import summary

import time

if __name__ == "__main__":

    #model = "mobilenet"             # ssd based in the mobilenet with standard convolutional layers
    model = "mobilenet_dw"         # ssd based in the mobilenet with depthwise convolutional layers
    #model = "vgg"                  # ssd based in the vgg16

    alpha = 1.0

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
    start_time = time.time()

    with torch.no_grad():
        for i in range(100):

            # generate a random image, predict over it, and count the time
            image = torch.randn(1, in_dim[0], in_dim[1], in_dim[2]).to(device)

            prediction = model(image)

            pred_period = min(time.time()-start_time, 2.0)
            prediction_time.add_value(pred_period)

            start_time = time.time()

            print("[{0}/{1}]".format(i,1000))
            print(pred_period)

    print("Average prediction time: {0:.4f}".format(prediction_time.get_average()))
    print("Min prediction time: {0:.4f}".format(prediction_time.min))
    print("Max prediction time: {0:.4f}".format(prediction_time.max))
    print("Sum prediction time: {0:.4f}".format(prediction_time.get_sum()))    
    
'''
Mobilenet DW alpha=1.0
Total params: 7,779,168
Average prediction time: 0.0579
Min prediction time: 0.0517
Max prediction time: 0.0631
Sum prediction time: 5.7851

Mobilenet DW alpha=0.75
Total params: 6,198,224
Average prediction time: 0.0555
Min prediction time: 0.0438
Max prediction time: 0.0609
Sum prediction time: 5.5483


Mobilenet DW alpha=0.5
Total params: 4,878,656
Average prediction time: 0.0577
Min prediction time: 0.0542
Max prediction time: 0.0607
Sum prediction time: 5.7737

Mobilenet Conv2d alpha=1.0
Total params: 24,475,200
Average prediction time: 0.0486
Min prediction time: 0.0392
Max prediction time: 0.0519
Sum prediction time: 4.8623

Mobilenet Conv2d alpha=0.75
Total params: 15,583,736
Average prediction time: 0.0485
Min prediction time: 0.0420
Max prediction time: 0.0509
Sum prediction time: 4.8530

Mobilenet Conv2d alpha=0.5
Total params: 9,044,656
Average prediction time: 0.0469
Min prediction time: 0.0440
Max prediction time: 0.0492
Sum prediction time: 4.6917

VGG16
Total params: 26,284,974
Average prediction time: 0.2929
Min prediction time: 0.0655
Max prediction time: 0.3057
Sum prediction time: 29.2857

'''
