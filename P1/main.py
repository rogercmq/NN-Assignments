from model import *
from optimizer import *
from dataloader import *
from loss import *
from metrics import draw
import math
from tqdm import tqdm


def get_config():
    config = {
        # model
        'input_channel': 28*28,
        'middle_channel': [128, 64, 64],
        'classes': 10,
        'init': 'xavier',             # random | xavier | zeros
        'dropout': -1,                # > 0. if used
        'l2_lamda': -1,               # > 0. if used
        'lr': 0.02,                   # initial LR
        'decay': 'step',              # step | cosine | cycle | const
        'epoch': 20,                  # 10 if regularization techniques are not implemented else 20. 
        'bs': 16,                     # batch size
        'loss': 'cross_entropy_l2',   # cross_entropy | cross_entropy_l2
        'test_interval': 1000,        # evaluate the accuracy of my MLP every test_interval iterations
        'output_dir': 'output',       # directory for saving logs
        'experiment_name': 'result',
    }
    return config


def train(train_set, test_set, model, config):
    bs = config['bs']
    iter_per_epoch = math.ceil(len(train_set)/bs)
    optimizer = get_optimizer(config, iter_per_epoch)
    loss_fn = get_loss(config)
    model.train()
    total_iter = 0

    iter_log = []
    train_accs = []
    test_accs = []

    for e in range(config['epoch']):
        iter = 0
        for i in range(iter_per_epoch):
            train_data = train_set[i*bs:min((i+1)*bs, len(train_set))]
            train_x = np.array([data[0] for data in train_data])
            train_y = np.array([data[1] for data in train_data])
            pred_y = model(train_x)
            loss = loss_fn(pred_y, train_y, model)
            optimizer.step(model, train_y)
            iter+=1
            total_iter+=1
            optimizer.lr_scheduler(iter)
            if total_iter % config['test_interval'] == 0:
                train_acc = test(train_set, model)
                test_acc = test(test_set, model)
                iter_log.append(total_iter)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
    return iter_log, train_accs, test_accs


def test(test_set, model):
    model.test()
    cnt = 0
    for i in tqdm(range(len(test_set))):
        data, label = test_set[i]
        data = np.expand_dims(data, axis=0)
        pred = model(data)
        pred = np.argmax(pred, axis=1)
        gt = np.argmax(label, axis=0)
        if pred == gt:
            cnt +=1        
    model.train()
    return (cnt / len(test_set))


def main():
    config = get_config()
    print(config)
    model = build_model(config)
    train_set, test_set = get_data()
    iter_log, train_accs, test_accs = train(train_set, test_set, model, config)
    draw(config, iter_log, train_accs, test_accs)
    print(f"Trainset Acc:{train_accs}\nTestset Acc:{test_accs}")


if __name__ == '__main__':
    main()
