import numpy as np
import math
import random
from mlp import MnistMLP


class SGDtrainer():
    def __init__(self, model:MnistMLP, cfg:dict):
        assert type(cfg) == type(dict())
        self.cfg = cfg
        self.mini_bs = cfg['mini_bs'] if 'mini_bs' in cfg.keys() else 1
        self.lr = cfg['lr'] if 'lr' in cfg.keys() else 1e-2
        self.num_iters = cfg['num_iters'] if 'num_iters' in cfg.keys() else 10000
        self.lr_strategy =  cfg['lr_strategy'] if 'lr_strategy' in cfg.keys() else 'cosine'
        self.model = model
        self.init_lr = self.lr
        self.cur_iter = 0
        self.stepsize = self.num_iters // 2


    def __call__(self, train_data):
        mini_batches = [train_data[k:k+self.mini_bs] for k in range(0, len(train_data), self.mini_bs)]
        self.cur_iter += 1
        for xys in mini_batches:
            self.__forward(xys)


    def __forward(self, xys):
        nabla_w = [np.zeros(w.shape) for w in self.model.weights]
        nabla_b = [np.zeros(b.shape) for b in self.model.biases]
        for x,y in xys:
            delta_nabla_w, delta_nabla_b = self.model.forward(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.__adjust_lr()
        self.model.weights = [w - (self.lr / len(xys)) * nw for w, nw in zip(self.model.weights, nabla_w)]
        self.model.biases =  [b - (self.lr / len(xys)) * nb for b, nb in zip(self.model.biases, nabla_b)]


    def __adjust_lr(self):
        if self.lr_strategy == 'step':
            if self.cur_iter % self.stepsize == 0:
                self.lr *= 0.5 
        elif self.lr_strategy == 'cosine':
            self.lr = 0.5 * (1 + math.cos(math.pi * self.cur_iter / self.num_iters)) * self.init_lr
        elif self.lr_strategy == 'cycle':
            max_lr = self.init_lr
            base_lr = self.init_lr * 0.01
            cycle = np.floor(1 + self.cur_iter / (2  * self.stepsize))
            x = np.abs(self.cur_iter / self.stepsize - 2 * cycle + 1)
            self.lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
