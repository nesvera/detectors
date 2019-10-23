import torch
from torch import nn
import torch.nn.functional as F

from utils import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiBoxLoss(nn.Module):

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super(MultiBoxLoss, self).__init__()

        self.priors_cxcy = priors_cxcy
        self.priors_xy = utils.cxcy_to_xy(priors_cxcy)
        
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.l1_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        true_locs = torch.zeros((batch_size, n_priors, 4),
                                dtype=torch.float).to(device)                   # [N, n_priors, 4]
        true_classes = torch.zeros((batch_size, n_priors),
                                dtype=torch.long).to(device)                    # [N, n_priors]

        # For each image of the batch
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = utils.find_jaccard_overlap(boxes[i], self.priors_xy)      # [n_objects, n_prios]

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

            # Get the labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]             # [n_priors]

            # Set as background all priors that have overlap less than overlap
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0   # [n_priors]

            # Append image ground truth to the batch ground truth
            true_classes[i] = label_for_each_prior

            # Encode coordinates from xmin,ymin,xmax,ymax to center-offset
            true_locs[i] = utils.cxcy_to_gcxgcy(utils.xy_to_cxcy(boxes[i][object_for_each_prior]),
                                                self.priors_cxcy)               # [n_priors, 4]

        # Identify priors taht are positive (object/non-background)
        positive_priors = true_classes != 0                                     # [N, n_priors]

        # Localization loss is computed only with positive (non-background) priors
        loc_loss = self.l1_loss(predicted_locs[positive_priors], true_locs[positive_priors])   # scalar value

        # Confidence loss is computed over positive priors and the most difficult negative priors in each image.
        # n_negative = neg_pos_ratio * n_positives
        # Take the n_negative priors with maximum loss
        # This is called Hard Negative Mining. It concentrates on hardest negatives in each image to minimize
        # the pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)                                # [N]
        n_hard_negatives = self.neg_pos_ratio*n_positives                       # [N]

        # Calculate the loss for all priors
        conf_loss_all = self.cross_entropy_loss(predicted_scores.view(-1, n_classes), 
                                                true_classes.view(-1))          # [N * n_priors]
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)                # [N, n_priors]

        # Get positive priors
        conf_loss_pos = conf_loss_all[positive_priors]                          # [n_positives]

        # Get hard-negative priors
        conf_loss_neg = conf_loss_all.clone()                                   # [N, n_priors]
        
        # ignore positive priors
        conf_loss_neg[positive_priors] = 0

        # sort based in the loss
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)

        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device) # [N,8732]
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_pos.sum() + conf_loss_hard_neg.sum())/n_positives.sum().float()

        # Final loss
        return (conf_loss + self.alpha*loc_loss)
