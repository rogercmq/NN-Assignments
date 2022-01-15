import math
import numpy as np

class SGDOptimizer():
    def __init__(self, config, iter_per_epoch):
        epoch = config['epoch']
        self.total_iter = epoch * iter_per_epoch
        self.decay = config['decay']
        self.lr = config['lr']
        self.init_lr = self.lr

    def step(self, model, gt_y):
        model.backward(self.lr, gt_y)

    def lr_scheduler(self, iter):
        stepsize = self.total_iter // 3
        if self.decay == 'const':
            pass
        elif self.decay == 'step':
            if iter % stepsize == 0:
                self.lr *= 0.1
        elif self.decay == 'cosine':
            self.lr = 0.5 * (1 + math.cos(math.pi * iter / self.total_iter)) * self.init_lr
        elif self.decay == 'cycle':
            max_lr = self.init_lr
            base_lr = self.init_lr * 0.1
            cycle = np.floor(1 + iter / (2  * stepsize))
            x = np.abs(iter / stepsize - 2 * cycle + 1)
            self.lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
        else:
            raise NotImplementedError


def get_optimizer(config, len_train_set):
    return SGDOptimizer(config, len_train_set)
