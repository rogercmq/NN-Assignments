from abc import ABC, abstractmethod
import numpy as np

def build_model(config):
    model = MLP(config)
    model.init_params(config['init'])
    return model

class MyModule(ABC):
    '''nn.Module'''
    def __init__(self):
        self.is_train = False
    
    
    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, lr, grad):
        pass

    def train(self):
        self.is_train = True

    def test(self):
        self.is_train = False

    def init_params(self, init):
        pass


class MLP(MyModule):
    '''mlp for number cls'''
    def __init__(self, config):
        super().__init__()
        self.input_channel = config['input_channel']
        self.middle_channel = config['middle_channel']
        self.classes = config['classes']
        dropout = config['dropout']
        lamda = config['l2_lamda'] if config['l2_lamda'] > 0. else None
        if dropout > 0.0:
            self.layers = [Linear(self.input_channel, self.middle_channel[0], lamda=lamda), Dropout(dropout), Sigmoid()]
        else:
            self.layers = [Linear(self.input_channel, self.middle_channel[0], lamda=lamda), Sigmoid()]
        for d in range(0, len(self.middle_channel)-1):
            self.layers.append(Linear(self.middle_channel[d], self.middle_channel[d+1], lamda=lamda))
            if dropout > 0.0:
                self.layers.append(Dropout(dropout))
            self.layers.append(Sigmoid())
        self.layers.append(Linear(self.middle_channel[-1], self.classes, lamda=lamda))
        self.layers.append(Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def backward(self, lr, gt_y):
        grad = gt_y
        for layer in reversed(self.layers):
            grad, _ = layer.backward(lr, grad)
        return

    def print_net(self):
        for _ in self.layers:
            print(_.weight.shape)
            print(_.bias.shape)

    def train(self):
        self.is_train = True
        for layer in self.layers:
            layer.train()

    def test(self):
        self.is_train = False
        for layer in self.layers:
            layer.test()
    
    def init_params(self, init):
        for layer in self.layers:
            layer.init_params(init)


class Linear(MyModule):
    def __init__(self, in_channel, out_channel, lamda=0.):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = np.zeros([in_channel, out_channel], dtype=np.float32)
        self.bias = np.zeros([out_channel], dtype=np.float32)
        self.lamda = lamda # L2 regularization

    def forward(self, x):
        '''
            input shape: [n, d]
            output shape: [n, d]
        '''
        if self.is_train:
            self.x = x
        x = np.matmul(x, self.weight) + self.bias
        return x

    def backward(self, lr, grad, l2_reg=False):
        grad_w = np.matmul(self.x.T, grad)
        grad_b = np.sum(grad, axis=0)
        if l2_reg:
            grad_w += self.weight * self.lamda
        grad = np.matmul(grad, self.weight.T)
        self.weight -= (lr * grad_w)
        return grad, None

    def init_params(self, init):
        if init == 'random':
            self.weight = np.random.normal(0, 0.02, (self.in_channel, self.out_channel))
            self.bias = np.zeros(self.out_channel, dtype=np.float32)
        elif init == 'xavier':
            std = np.sqrt(2. / (self.in_channel + self.out_channel))
            self.weight = np.random.normal(loc=0., scale=std, size=[self.in_channel, self.out_channel])
            self.bias = np.random.normal(loc=0., scale=std, size=[self.out_channel])
        elif init == 'zeros':
            self.weight = np.zeros((self.in_channel, self.out_channel))
            self.bias = np.zeros(self.out_channel)
        else:
            raise NotImplementedError('init method not implemented')


class Softmax(MyModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
            input shape: [n, d]
            output shape: [n, d]
        '''
	# stable softmax
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
        if self.is_train:
            self.softmax_x = softmax_x
        return softmax_x
 
    def backward(self, lr, gt_y):
        # 使用CE Loss，反向传播过程简化
        grad = self.softmax_x - gt_y
        return grad, None


class Sigmoid(MyModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = 1/(1 + np.exp(-x))
        if self.is_train:
            self.x = x
        return x
        
    def backward(self, lr, grad):  
        grad = grad * self.x * (1-self.x)
        return grad, None


class Dropout(MyModule):
    def __init__(self, p=0.2):
        self.p = p
        self._mask = None

    def forward(self, X, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward(self, lr, grad):
        return grad * self._mask, None
