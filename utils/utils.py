import torch

class Average():
    
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add_value(self, value):
        self.sum += value
        self.count += 1

    def get_size(self):
        return self.count

    def get_average(self):
        if self.count > 0:
            self.avg = float(self.sum)/self.count
            return self.avg
        else:
            return 0

def xy_to_cxcy(boxes_xy):
    """
    Convert bounding boxes from boundary coordinates (xmin, ymin, xmax, ymax)
    to center-size coordinates (c_x, c_y, w, h).

        Parameters
        ----------
        boxes_xy : tensor
            Bounding boxes in boundary coordinates 
            (n_boxes, 4)

        Returns
        -------
        tensor
            Bounding boxes in center-size coordinates 
            (n_boxes, 4)
    """

    return torch.cat([(boxes_xy[:, 2:] + boxes_xy[:, :2])/2,
                      (boxes_xy[:, 2:] - boxes_xy[:, :2])], 1)

def cxcy_to_xy(boxes_cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h)
    to boundary coordinates (xmin, ymin, xmax, ymax).

        Parameters
        ----------
        boxes_cxcy : tensor
            Bounding boxes in center-size coordinates
            (n_boxes, 4)

        Returns
        -------
        tensor
            Bounding boxes in boundary coordinates 
            (n_boxes, 4)
    """
    return torch.cat([(boxes_cxcy[:, :2] - (boxes_cxcy[:, 2:]/2)),
                      (boxes_cxcy[:, :2] + (boxes_cxcy[:, 2:]/2))], 1)

def cxcy_to_gcxgcy(boxes_cxcy, priors_cxcy):
    """
        Encode bounding boxes from center-size coordinates (c_x, c_y, w, h)
        w.r.t the corresponding prior boxes.

        For the center coordinates, find the offset w.r.t the prior box, and scale
        by the size of the prior box.

        for the size coordinate, scale by the size of the prior box, and corvert to
        the log-space.
    
        Parameters
        ----------
        boxes_cxcy : tensor
            Bounding boxes in center-size coordinates
            (n_boxes, 4)

        priors_cxcy : tensor
            Prior boxes in center-size coordinates
            (n_boxes, 4)

        Returns
        -------
        tensor
            Encoded bouding boxes
            (n_boxes, 4)
    """

    center_variance = 0.1
    size_variance = 0.2

    return torch.cat([(((boxes_cxcy[:, :2] - priors_cxcy[:, :2])/priors_cxcy[:, 2:])*center_variance),
                      (torch.log(boxes_cxcy[:, 2:] / priors_cxcy[:, 2:])*size_variance)], 1)

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
        Decode bounding boxes predicted by the model to center-size coordinates 
        w.r.t the corresponding prior boxes.
    
        Parameters
        ----------
        gcxgcy : tensor
            Bouding boxes predicted by the model in prior-offset-coordinate
            (n_boxes, 4)

        priors_cxcy : tensor
            Prior boxes in center-size coordinates
            (n_boxes, 4)

        Returns
        -------
        tensor
            Decoded bouding boxes
            (n_boxes, 4)
    """

    center_variance = 0.1
    size_variance = 0.2

    return torch.cat([((gcxgcy[:, :2]*priors_cxcy[:, 2:]/center_variance) + priors_cxcy[:, :2]),
                      (torch.exp(gcxgcy[:, 2:]/size_variance)*priors_cxcy[:, 2:])], 1)

def find_jaccard_overlap(set_1, set_2):
    """
        Calculate the Jaccard Overlap (IoU) of every combination between two sets of boxes that
        are in boundary coordinates.
    
        Parameters
        ----------
        set_1 : tensor
            Bounding boxes in boundary coordinates
            (n1, 4)

        set_2 : tensor
            Bounding boxes in boundary coordinates
            (n2, 4)

        Returns
        -------
        tensor
            Encoded bouding boxes
            (n1, n2)
    """

    # Find intersections
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # [n1, n2, 2]
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # [n1, n2, 2]
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)             # [n1, n2, 2]
    
    intersection = intersection_dims[:,:,0] * intersection_dims[:,:,1]              # [n1, n2]

    # Find areas
    area_set_1 = (set_1[:,2] - set_1[:,0])*(set_1[:,3] - set_1[:,1])                # [n1]    
    area_set_2 = (set_2[:,2] - set_2[:,0])*(set_2[:,3] - set_2[:,1])                # [n2]

    # Find union
    union = area_set_1.unsqueeze(1) + area_set_2.unsqueeze(0) - intersection        # [n1, n2]

    return (intersection / union)