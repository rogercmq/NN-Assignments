import numpy as np
from tqdm import tqdm
import random 

from mlp import MnistMLP
from dataloader import get_data
from optimizer import SGDtrainer


def train_net(dataloader, optimizer:SGDtrainer, batch_size, total_iters):
    for i in tqdm(range(total_iters)):
        idx = i % len(dataloader)
        train_xys = dataloader[idx: idx+batch_size] # 长度为 batch_size 的 tuple 列表
        assert type(train_xys) == type(list()) and type(train_xys[-1]) == type(tuple())
        assert len(train_xys[-1][0]) == 28*28 and len(train_xys[-1][1]) == 10
        optimizer(train_xys)
    return optimizer.model


def evaluate_net(model:MnistMLP, dataloader):
    results = [(np.argmax(model.forward(x)), np.argmax(y)) for (x, y) in dataloader]
    print(results)
    return sum(int(x == y) for (x,y) in results) / len(dataloader)


if __name__ == '__main__':
    ITERATIONS = 10000
    LR = 1e-5
    LR_STRATEGY = 'cosine'
    ACTIVATION_TYPE = 'relu'
    INIT_METHOD = 'random'
    HIDDEN_SIZE = 64
    NUM_CLASSES = 10
    INPUT_SHAPE = 28
    EPOCHES = 10
    print("Collecting data...")
    trainset, testset = get_data()
    assert len(trainset) == 60000 and len(testset) == 10000
    assert type(trainset[0]) == type(tuple())
    assert type(trainset[0][0]) == type(np.array(1.0))
    random.shuffle(trainset)
    random.shuffle(testset)
    batch_size = len(trainset) * EPOCHES // ITERATIONS
    mlp = MnistMLP(size=[INPUT_SHAPE*INPUT_SHAPE, HIDDEN_SIZE, NUM_CLASSES], 
                   act_type=ACTIVATION_TYPE, 
                   init_type=INIT_METHOD)
    print(mlp)
    optimizer = SGDtrainer(model=mlp, cfg=dict(mini_bs=10, 
                                               lr=LR,
                                               num_iters=ITERATIONS,
                                               lr_strategy=LR_STRATEGY))
    # print("Start testing before training...")
    # accuracy = evaluate_net(model=mlp, dataloader=testset)
    # print(accuracy)
    print("Start training...")
    mlp = train_net(dataloader=trainset, 
                    optimizer=optimizer,
                    batch_size=batch_size,
                    total_iters=ITERATIONS)
    print("Start testing...")
    accuracy = evaluate_net(model=mlp, dataloader=testset)
    print(accuracy)
