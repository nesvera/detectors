import torch
from torch import nn
import torch.nn.functional as F

from math import ceil, sqrt

from utils import utils

#from classifiers.models import MobileNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv2dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv2dBn, self).__init__()
        
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        return out

class MobileNetV1Conv224(nn.Module):
    """
    Model based on the MobileNet Architecture described in the paper 
    "MobileNets: Efficient Convolutional Neural Networks for Mobile 
    Vision Applications" using standard convolutional layers due to
    the speed problems w.r.t the depthwise separable layers

    Number of parameters (alpha=0.5): 
    ...

    Attributes
    ----------
    pretrained_weights : state_dict
        dictionary of the model trained as a classifier
    alpha : float
        multiplier for the number of channels for each conv layer

    Methods
    -------
    foward(sound=None)
        Method used by the pytorch API, that return some feature maps
    """

    def __init__(self, pretrained_weights, alpha=0.5):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """

        super(MobileNetV1Conv224, self).__init__()

        self.alpha = alpha

        # Input tensor
        # [N, 3, 224, 224]
        
        self.conv_1 = Conv2dBn(3, 
                                ceil(self.alpha*32), 
                                kernel_size=3, padding=1, stride=2)
        # [N, 32, 112, 112]

        self.conv_2 = Conv2dBn(ceil(self.alpha*32), 
                                ceil(self.alpha*64), 
                                kernel_size=3, padding=1, stride=1)
        # [N, 64, 112, 112]

        self.conv_3 = Conv2dBn(ceil(self.alpha*64), 
                                ceil(self.alpha*128), 
                                kernel_size=3, padding=1, stride=2)
        # [N, 128, 56, 56]

        self.conv_4 = Conv2dBn(ceil(self.alpha*128),
                                ceil(self.alpha*128),
                                kernel_size=3, padding=1, stride=1)
        # [N, 128, 56, 56]

        self.conv_5 = Conv2dBn(ceil(self.alpha*128),
                                ceil(self.alpha*256),
                                kernel_size=3, padding=1, stride=2)
        # [N, 256, 28, 28]

        self.conv_6 = Conv2dBn(ceil(self.alpha*256),
                                ceil(self.alpha*256),
                                kernel_size=3, padding=1, stride=1)
        # [N, 256, 28, 28]

        self.conv_7 = Conv2dBn(ceil(self.alpha*256),
                                ceil(self.alpha*512),
                                kernel_size=3, padding=1, stride=2)
        # [N, 512, 14, 14]

        self.conv_8 = Conv2dBn(ceil(self.alpha*512),
                                ceil(self.alpha*512),
                                kernel_size=3, padding=1, stride=1)
        # [N, 512, 14, 14]
        # repeated 5 times

        self.conv_9 = Conv2dBn(ceil(self.alpha*512),
                                ceil(self.alpha*1024),
                                kernel_size=3, padding=1, stride=2)
        # [N, 1024, 7, 7]

        self.conv_10 = Conv2dBn(ceil(self.alpha*1024),
                                 ceil(self.alpha*1024),
                                 kernel_size=3, padding=4, stride=2)
        # [N, 1024, 7, 7]

        if pretrained_weights is not None:
            self.load_pretrained_layers(pretrained_weights)

    def load_pretrained_layers(self, pretrained_weights):
        
        # current state of the weights
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # pretrained base
        pretrained_state_dict = pretrained_weights
        pretrained_param_names = list(pretrained_state_dict.keys())

        # transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

    def forward(self, image):
        """
        Forward propagation.

        Parameters
        ----------
        image : tensor
            Images, a tensor of dimensions (N, 3, 224, 224)
        
        Returns
        -------
        out_name : tensor
            Predictions from 
        """
                                                    # [N, 3,     224, 224]
        out = self.conv_1(image)                    # [N, a*32,  112, 112]
        out = self.conv_2(out)                      # [N, a*64,  112, 112]
        out = self.conv_3(out)                      # [N, a*128, 56,  56]
        out = self.conv_4(out)                      # [N, a*128, 56,  56]
        conv_d56 = out

        out = self.conv_5(out)                      # [N, a*256, 28,  28]
        out = self.conv_6(out)                      # [N, a*256, 28,  28]
        conv_d28 = out
        
        out = self.conv_7(out)                      # [N, a*512, 14,  14]
        out = self.conv_8(out)                      # [N, a*512, 14, 14]
        conv_d14 = out

        out = self.conv_9(out)                      # [N, a*1024, 7, 7]
        out = self.conv_10(out)                     # [N, a*1024, 7, 7]
        conv_d7 = out

        return conv_d56, conv_d28, conv_d14, conv_d7

class AuxiliaryConvolutions(nn.Module):

    def __init__(self, alpha=0.5):
        super(AuxiliaryConvolutions, self).__init__()

        self.alpha = alpha

        # [N, 1024, 7, 7]

        # add auxiliary convolutions
        self.aux_conv_1_1 = Conv2dBn(512,
                                     256,
                                     kernel_size=1, padding=0, stride=1)
        # [N, 256, 7, 7]

        self.aux_conv_1_2 = Conv2dBn(256,
                                     512,
                                     kernel_size=3, padding=0, stride=1)
        # [N, 512, 5, 5]

        self.aux_conv_2_1 = Conv2dBn(512,
                                     256,
                                     kernel_size=1, padding=0, stride=1)
        # [N, 256, 5, 5]

        self.aux_conv_2_2 = Conv2dBn(256,
                                     256,
                                     kernel_size=3, padding=0, stride=1)
        # [N, 256, 3, 3]

        self.aux_conv_3_1 = Conv2dBn(256,
                                     128,
                                     kernel_size=1, padding=0, stride=1)
        # [N, 128, 3, 3]

        self.aux_conv_3_2 = Conv2dBn(128,
                                     256,
                                     kernel_size=3, padding=0, stride=1)
        # [N, 256, 1, 1]                                             

    def forward(self, conv_d7_feats):
        
        out = self.aux_conv_1_1(conv_d7_feats)      # [N, 256, 7, 7]
        out = self.aux_conv_1_2(out)                # [N, 512, 5, 5]
        conv_d5 = out

        out = self.aux_conv_2_1(out)                # [N, 256, 5, 5]
        out = self.aux_conv_2_2(out)                # [N, 256, 3, 3]
        conv_d3 = out

        out = self.aux_conv_3_1(out)                # [N, 128, 3, 3]
        out = self.aux_conv_3_2(out)                # [N, 256, 1, 1]
        conv_d1 = out

        return conv_d5, conv_d3, conv_d1

class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # number of prior boxes per position in each feature map
        n_boxes = {'conv_d56': 2,
                   'conv_d28': 4,
                   'conv_d14': 6,
                   'conv_d7':  6,
                   'conv_d5':  6,
                   'conv_d3':  4,
                   'conv_d1':  4}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv_d56 = nn.Conv2d(64, n_boxes['conv_d56']*4, kernel_size=3, padding=1)
        self.loc_conv_d28 = nn.Conv2d(128, n_boxes['conv_d28']*4, kernel_size=3, padding=1)
        self.loc_conv_d14 = nn.Conv2d(256, n_boxes['conv_d14']*4, kernel_size=3, padding=1)
        self.loc_conv_d7  = nn.Conv2d(512, n_boxes['conv_d7']*4,  kernel_size=3, padding=1)
        self.loc_conv_d5  = nn.Conv2d(512, n_boxes['conv_d5']*4,  kernel_size=3, padding=1)
        self.loc_conv_d3  = nn.Conv2d(256, n_boxes['conv_d3']*4,  kernel_size=3, padding=1)
        self.loc_conv_d1  = nn.Conv2d(256, n_boxes['conv_d1']*4,  kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv_d56 = nn.Conv2d(64, n_boxes['conv_d56']*n_classes, kernel_size=3, padding=1)
        self.cl_conv_d28 = nn.Conv2d(128, n_boxes['conv_d28']*n_classes, kernel_size=3, padding=1)
        self.cl_conv_d14 = nn.Conv2d(256, n_boxes['conv_d14']*n_classes, kernel_size=3, padding=1)
        self.cl_conv_d7 =  nn.Conv2d(512, n_boxes['conv_d7']*n_classes,  kernel_size=3, padding=1)
        self.cl_conv_d5 =  nn.Conv2d(512, n_boxes['conv_d5']*n_classes,  kernel_size=3, padding=1)
        self.cl_conv_d3 =  nn.Conv2d(256, n_boxes['conv_d3']*n_classes,  kernel_size=3, padding=1)
        self.cl_conv_d1 =  nn.Conv2d(256, n_boxes['conv_d1']*n_classes,  kernel_size=3, padding=1)

        self.loc_activation = nn.Hardtanh()

    def forward(self, conv_d56, conv_d28, conv_d14, conv_d7, conv_d5, conv_d3, conv_d1):

        batch_size = conv_d56.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        loc_pred_conv_d56 = self.loc_activation(self.loc_conv_d56(conv_d56))    # [N, b*4, 56, 56]
        loc_pred_conv_d56 = loc_pred_conv_d56.permute(0, 2, 3, 1).contiguous()  # [N, 56, 56, b*4]
        loc_pred_conv_d56 = loc_pred_conv_d56.view(batch_size, -1, 4)           # [N, 6272, 4]


        loc_pred_conv_d28 = self.loc_activation(self.loc_conv_d28(conv_d28))    # [N, b*4, 28, 28]
        loc_pred_conv_d28 = loc_pred_conv_d28.permute(0, 2, 3, 1).contiguous()  # [N, 28, 28, b*4]
        loc_pred_conv_d28 = loc_pred_conv_d28.view(batch_size, -1, 4)           # [N, 4704, 4]

        loc_pred_conv_d14 = self.loc_activation(self.loc_conv_d14(conv_d14))    # [N, b*4, 14, 14]
        loc_pred_conv_d14 = loc_pred_conv_d14.permute(0, 2, 3, 1).contiguous()  # [N, 14, 14, b*4]
        loc_pred_conv_d14 = loc_pred_conv_d14.view(batch_size, -1, 4)           # [N, 1176, 4]

        loc_pred_conv_d7 = self.loc_activation(self.loc_conv_d7(conv_d7))       # [N, b*4, 7, 7]
        loc_pred_conv_d7 = loc_pred_conv_d7.permute(0, 2, 3, 1).contiguous()    # [N, 7, 7, b*4]
        loc_pred_conv_d7 = loc_pred_conv_d7.view(batch_size, -1, 4)             # [N, 294, 4]

        loc_pred_conv_d5 = self.loc_activation(self.loc_conv_d5(conv_d5))       # [N, b*4, 5, 5]
        loc_pred_conv_d5 = loc_pred_conv_d5.permute(0, 2, 3, 1).contiguous()    # [N, 5, 5, b*4]
        loc_pred_conv_d5 = loc_pred_conv_d5.view(batch_size, -1, 4)             # [N, 150, 4]

        loc_pred_conv_d3 = self.loc_activation(self.loc_conv_d3(conv_d3))       # [N, b*4, 3, 3]
        loc_pred_conv_d3 = loc_pred_conv_d3.permute(0, 2, 3, 1).contiguous()    # [N, 3, 3, b*4]
        loc_pred_conv_d3 = loc_pred_conv_d3.view(batch_size, -1, 4)             # [N, 36, 4]

        loc_pred_conv_d1 = self.loc_activation(self.loc_conv_d1(conv_d1))       # [N, b*4, 1, 1]
        loc_pred_conv_d1 = loc_pred_conv_d1.permute(0, 2, 3, 1).contiguous()    # [N, 1, 1, b*4]
        loc_pred_conv_d1 = loc_pred_conv_d1.view(batch_size, -1, 4)             # [N, b*4, 4]

        locs = torch.cat([loc_pred_conv_d56,
                          loc_pred_conv_d28,
                          loc_pred_conv_d14,
                          loc_pred_conv_d7,
                          loc_pred_conv_d5,
                          loc_pred_conv_d3,
                          loc_pred_conv_d1], dim=1)                             # [N, 12633, 4]

        # Predict classes in localization boxes
        cl_pred_conv_d56 = self.cl_conv_d56(conv_d56)                               # [N, b*n_classes, 56, 56]
        cl_pred_conv_d56 = cl_pred_conv_d56.permute(0, 2, 3, 1).contiguous()        # [N, 56, 56, b*n_classes]  
        cl_pred_conv_d56 = cl_pred_conv_d56.view(batch_size, -1, self.n_classes)    # [N, 6272, 30]

        cl_pred_conv_d28 = self.cl_conv_d28(conv_d28)                               # [N, b*n_classes, 28, 28]
        cl_pred_conv_d28 = cl_pred_conv_d28.permute(0, 2, 3, 1).contiguous()        # [N, 28, 28, b*n_classes]  
        cl_pred_conv_d28 = cl_pred_conv_d28.view(batch_size, -1, self.n_classes)    # [N, 4704, 30]  

        cl_pred_conv_d14 = self.cl_conv_d14(conv_d14)                               # [N, b*n_classes, 14, 14]
        cl_pred_conv_d14 = cl_pred_conv_d14.permute(0, 2, 3, 1).contiguous()        # [N, 14, 14, b*n_classes]  
        cl_pred_conv_d14 = cl_pred_conv_d14.view(batch_size, -1, self.n_classes)    # [N, 1176, 30]  

        cl_pred_conv_d5 = self.cl_conv_d5(conv_d5)                                  # [N, b*n_classes, 5, 5]
        cl_pred_conv_d5 = cl_pred_conv_d5.permute(0, 2, 3, 1).contiguous()          # [N, 5, 5, b*n_classes]  
        cl_pred_conv_d5 = cl_pred_conv_d5.view(batch_size, -1, self.n_classes)      # [N, 294, 30]

        cl_pred_conv_d7 = self.cl_conv_d7(conv_d7)                                  # [N, b*n_classes, 7, 7]
        cl_pred_conv_d7 = cl_pred_conv_d7.permute(0, 2, 3, 1).contiguous()          # [N, 7, 7, b*n_classes]  
        cl_pred_conv_d7 = cl_pred_conv_d7.view(batch_size, -1, self.n_classes)      # [N, 150, 30]

        cl_pred_conv_d3 = self.cl_conv_d3(conv_d3)                                  # [N, b*n_classes, 3, 3]
        cl_pred_conv_d3 = cl_pred_conv_d3.permute(0, 2, 3, 1).contiguous()          # [N, 3, 3, b*n_classes]  
        cl_pred_conv_d3 = cl_pred_conv_d3.view(batch_size, -1, self.n_classes)      # [N, 36, 30]

        cl_pred_conv_d1 = self.cl_conv_d1(conv_d1)                                  # [N, b*n_classes, 1, 1]
        cl_pred_conv_d1 = cl_pred_conv_d1.permute(0, 2, 3, 1).contiguous()          # [N, 1, 1, b*n_classes]  
        cl_pred_conv_d1 = cl_pred_conv_d1.view(batch_size, -1, self.n_classes)      # [N, 1, 30]

        class_score = torch.cat([cl_pred_conv_d56,
                                 cl_pred_conv_d28,
                                 cl_pred_conv_d14,
                                 cl_pred_conv_d7,
                                 cl_pred_conv_d5,
                                 cl_pred_conv_d3,
                                 cl_pred_conv_d1], dim=1)                          # [N, 12483, 30]

        return locs, class_score

class SSDMobileNet(nn.Module):

    def __init__(self, base_pretrained, num_classes):
        super(SSDMobileNet, self).__init__()

        if num_classes <= 0:
            print("Error: num_class must be a positive number")
            exit(1)

        self.num_classes = num_classes

        self.base_model = MobileNetV1Conv224(base_pretrained, alpha=0.5)
        self.aux_convs = AuxiliaryConvolutions(alpha=0.5)
        self.pred_convs = PredictionConvolutions(num_classes)

        self.priors_cxcy = self.create_prior()

    def forward(self, image):
        
        conv_d56, conv_d28, conv_d14, conv_d7 = self.base_model(image)
        
        conv_d5, conv_d3, conv_d1 = self.aux_convs(conv_d7)

        """ TODO: talvez tenha que reduzir o número de canais nas camadas auxiliares """

        locs, class_score = self.pred_convs(conv_d56,
                                            conv_d28,
                                            conv_d14,
                                            conv_d7,
                                            conv_d5,
                                            conv_d3,
                                            conv_d1)
    
        return locs, class_score

    def create_prior(self):

        n_boxes = {'conv_d56': 2,
                   'conv_d28': 4,
                   'conv_d14': 6,
                   'conv_d7':  6,
                   'conv_d5':  6,
                   'conv_d3':  4,
                   'conv_d1':  4}

        # dimensions of the feature maps
        feat_map_dims = {'conv_d56': 56,
                         'conv_d28': 28,
                         'conv_d14': 14,
                         'conv_d7':  7,
                         'conv_d5':  5,
                         'conv_d3':  3,
                         'conv_d1':  1}

        # object scale w.r.t the size of the image
        obj_scale = {'conv_d56': 0.2,
                     'conv_d28': 0.325,
                     'conv_d14': 0.35,
                     'conv_d7':  0.375,
                     'conv_d5':  0.55,
                     'conv_d3':  0.725,
                     'conv_d1':  0.9}
        
        # aspect ratio for the priors
        aspect_ratios = {'conv_d56': [1.],
                         'conv_d28': [1., 2., 0.5],
                         'conv_d14': [1., 2., 3., 0.5, .333],
                         'conv_d7':  [1., 2., 3., 0.5, .333],
                         'conv_d5':  [1., 2., 3., 0.5, .333],
                         'conv_d3':  [1., 2., 0.5],
                         'conv_d1':  [1., 2., 0.5]}

        feat_maps = list(feat_map_dims.keys())
        priors_boxes = []

        for k, fmap in enumerate(feat_maps):
            for i in range(feat_map_dims[fmap]):
                for j in range(feat_map_dims[fmap]):

                    cx = (j + 0.5)/feat_map_dims[fmap]
                    cy = (i + 0.5)/feat_map_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        priors_boxes.append([cx, cy,
                                             obj_scale[fmap]*sqrt(ratio),
                                             obj_scale[fmap]/sqrt(ratio)])

                        '''
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scale[fmap]*obj_scale[feat_maps[k+1]])
                            except IndexError:
                                additional_scale = 1.0
                            
                            priors_boxes.append([cx, cy,
                                                 additional_scale,
                                                 additional_scale])
                        '''

        priors_boxes = torch.FloatTensor(priors_boxes).to(device)
        priors_boxes.clamp_(0,1)

        return priors_boxes

    def detect_objects(self, pred_locs, pred_score, min_score, max_overlap, top_k):

        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)

        pred_score = F.softmax(pred_score, dim=2)                   # [N, n_priors, n_classes]

        # lists to store predictions for all images
        batch_boxes = list()
        batch_labels = list()
        batch_scores = list()

        # for each image of the batch
        for i in range(batch_size):

            # Decoded from regression format to bounding box format
            decoded_locs = utils.gcxgcy_to_cxcy(pred_locs[i], self.priors_cxcy)
            decoded_locs = utils.cxcy_to_xy(decoded_locs)

            # lists to store predictions for an image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # for each class
            for c in range(1, self.num_classes):

                # keep only predictions with scores above threshold
                class_scores = pred_score[i][:, c]                      # [n_priors] (FloatTensor)
                score_above_min = class_scores > min_score              # [n_priors] (BoolTensor)
                n_score_above_min = score_above_min.sum().item()

                if n_score_above_min == 0:
                    continue

                class_scores = class_scores[score_above_min]            # [n_qualified] (n_qualified <= n_priors)
                class_decoded_locs = decoded_locs[score_above_min]      # [n_qualified, 4]
    
                # Find overlap between each predicted box
                overlap = utils.find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                # Non-maximum supression 
                suppress = torch.zeros((n_score_above_min), dtype=torch.bool).to(device)

                for box in range(class_decoded_locs.size(0)):

                    # 
                    if suppress[box] == 1:
                        continue

                    suppress = suppress | (overlap[box] > max_overlap)

                    suppress[box] = False

                image_boxes.append(class_decoded_locs)

            # if there is no object, assign background for the image
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0, 0, 1, 1]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0]).to(device))


            # Create a single tensor
            image_boxes = torch.cat(image_boxes, dim=0)
            #image_labels = torch.cat(image_labels, dim=0)
            #image_scores = torch.cat(image_scores, dim=0)

            batch_boxes.append(image_boxes)
            #batch_labels.append(image_labels)
            #batch_scores.append(image_scores)
        

        return batch_boxes, 0, 0











