import torch
from torchvision.ops import nms

from ..box_utils import decode


class Detect():
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]

    def __call__(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
        Return:
            (tensor) Shape: [batch,num_classes,top_k,5], each row is [score, xmin, ymin, xmax, ymax]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5, device=loc_data.device)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i]

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                boxes = decoded_boxes[c_mask]
                # as in the original ssd.pytorch nms, only the top_k scoring boxes are candidates
                candidates = scores.topk(min(self.top_k, scores.size(0))).indices
                boxes, scores = boxes[candidates], scores[candidates]
                # idx of highest scoring and non-overlapping boxes per class
                ids = nms(boxes, scores, self.nms_thresh)
                count = ids.size(0)
                output[i, cl, :count] = torch.cat((scores[ids].unsqueeze(1), boxes[ids]), 1)
        return output
