import numpy as np


class MnistMLP():
    """ multi-layer MLP for MNIST classification """
    def __init__(self, size, act_type, init_type):
        """
            activation function: relu | sigmoid
            param init method: random | xavier
        """
        if isinstance(size, list):
            assert size[0] == 28*28 and size[-1] == 10
        else:
            raise ValueError('MLP depth init error.')
        if isinstance(act_type, str):
            assert act_type == 'relu' or act_type == 'sigmoid'
        else:
            raise ValueError('MLP activation function init error.')
        if isinstance(init_type, str):
            assert init_type == 'random' or init_type == 'xavier'
        else:
            raise ValueError('MLP param init_type init error.')

        self.depths = np.array(size)
        self.act_func_type = act_type
        self.weights = None
        self.biases = None
        self.init_params(init_type)
        self.debug_log = 0


    def __repr__(self):
        return f'weight matrix shape: {[w.shape for w in self.weights]}\n' \
               f'bias matrix shape: {[b.shape for b in self.biases]}'


    def init_params(self, init_type): # param initialization # 784 512 10
        if init_type == 'random': # standard gaussian
            self.weights = [np.random.normal(0, np.sqrt(2/y), (y, x)) for x,y in zip(self.depths[:-1], self.depths[1:])]
            self.biases = [np.random.normal(0, np.sqrt(2/y), (y, 1)) for y in self.depths[1:]]
        else:
            raise NotImplementedError


    def forward(self, x, y=None):
        """
            x.shape --> (784,)
            type(x) --> <class 'numpy.ndarray'>
            y.shape --> (10,)
        """
        x = x.reshape(-1, 1)
        if y is not None:
            y = y.reshape(-1, 1)
        if y is None: # forward only, for evaluating
            for i in range(len(self.weights)):
                x = np.dot(self.weights[i], x) + self.biases[i]
                x = self.activation(x)
            return x
        else: # forward & backward, for training
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            activation = x
            activations = [x]
            z_vector = []
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, activation) + b
                z_vector.append(z)
                activation = self.activation(z)
                activations.append(activation)
            loss = self.loss(activations[-1], y, return_grad=True)
            delta = loss * self.activation(z_vector[-1], bp=True)
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            nabla_b[-1] = delta
            for layer in range(2, len(self.depths)):
                z = z_vector[-layer]
                Rp = self.activation(z, bp=True)
                delta = np.dot(self.weights[-layer+1].transpose(), delta) * Rp
                nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
                nabla_b[-layer] = delta
            if self.debug_log < 1:
                # print(loss.squeeze(-1))
                # print(delta.squeeze(-1))
                #for item_ in nabla_w:
                #    print(item_.shape, item_)
                #print("!!")
                for item_ in activations:
                    print(item_.shape)
                self.debug_log += 1
            return (nabla_w, nabla_b)


    def activation(self, x, bp=False): # calculate activations & derivatives of activation funcs
        if bp == False: # forward
            if self.act_func_type == 'sigmoid':
                return 1.0 / (1.0 + np.exp(-x))
            elif self.act_func_type == 'relu':
                return (abs(x) + x) / 2
            else:
                raise NotImplementedError
        else: # backward
            if self.act_func_type == 'sigmoid':
                return self.activation(x) * (1 - self.activation(x))
            elif self.act_func_type == 'relu':
                return np.where(x > 0, 1, 0)
            else:
                raise NotImplementedError


    def loss(self, pred, gt, return_grad=True):
        if return_grad:
            return pred - gt  # softmax-交叉熵求导 https://zhuanlan.zhihu.com/p/60042105
        else:
            raise NotImplementedError



if __name__ == '__main__':
    model = MnistMLP(size=[784,64,10], act_type='relu', init_type='random')
