import numpy as np
from model import MLP, Linear

def get_loss(config):
    loss_type = config['loss']
    if loss_type == 'cross_entropy':
        return CrossEntropyLoss()
    elif loss_type == 'cross_entropy_l2':
        return L2Loss(config['l2_lamda'])
    else:
        raise NotImplementedError('loss function not implemented')


class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, pred_y, gt_y, model):
        log_y = np.log(pred_y+1e-10)
        loss = - gt_y * log_y
        loss = np.mean(np.sum(loss, axis=1))
        return loss


class L2Loss():
    def __init__(self, lamba):
        self.lamba = lamba
    def __call__(self, pred_y, gt_y, model:MLP):
        # CE Loss
        log_y = np.log(pred_y+1e-10)
        loss = - gt_y * log_y
        loss = np.mean(np.sum(loss, axis=1))
        # L2 Loss
        if self.lamba is not None:
            weight_sum = 0.
            for layer in model.layers:
                if isinstance(layer, Linear):
                    weight_sum += np.sum(np.square(layer.weight))
            loss += (1. * self.lamba / 2) * weight_sum
        return loss
