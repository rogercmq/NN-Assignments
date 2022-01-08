import csv
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models import LinearPredictor, LSTMPredictor, GRUPredictor
from dataloader import Trainer


def train_net(model, dataloader, optimizer, criterion, total_iter=10000, log_iter=1000, ckpt_iter=1000, model_name='model', model_path='./ckpts'):
    cur_iter = 0
    for _ in range(10000):
        for _, (batch_x, batch_y) in enumerate (dataloader): 
            outputs = model(batch_x)
            optimizer.zero_grad()   
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            cur_iter += 1
            if cur_iter % log_iter == 0:
                print(f"iter: {cur_iter}, loss: %1.5f" % loss.item())
            if cur_iter % ckpt_iter == 0 and cur_iter > 0:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(model.state_dict(), os.path.join(model_path,f'{model_name}_iter{cur_iter}.pt'))
            if cur_iter == total_iter:
                return


@torch.no_grad()
def evaluate_net(model, dataloader, criterion, save_path, save_name='result.csv'):
    results = None
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    total_iters = 0
    total_loss = 0
    for _, (batch_x, batch_y) in enumerate (dataloader): 
        outputs = model(batch_x)
        total_loss += criterion(outputs, batch_y)
        total_iters += 1
        pred = outputs.detach().numpy()
        if results is None:
            results = pred
        else:
            results = np.vstack((results, pred))
    avg_loss = total_loss / total_iters
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pd.DataFrame(results).to_csv(os.path.join(save_path, save_name), header=None, index=None)
    """
        预测结果保存成 csv，用于计算sMAPE
        示例:
            pd.DataFrame(np_array).to_csv("path/to/file.csv")
            df=pd.read_csv('myfile.csv', sep=',', header=None)
            df.values
    """
    return avg_loss


SEQ_LENGTH = 56
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LR = 0.0001
BATCH_SIZE = 64
ITERATION = 20000
CKPT_ITER = 2000
MODEL_TYPE = 'LSTM' # GRU | LSTM


if __name__ == '__main__':
    trainer = Trainer(seq_length=SEQ_LENGTH)
    if os.path.exists(f'data{SEQ_LENGTH}'):
        trainer.collect_data(load_npy=True, load_npy_path=f'data{SEQ_LENGTH}')
    else:
        trainer.collect_data(load_npy=False, load_npy_path=f'data{SEQ_LENGTH}')
    train_loader = torch.utils.data.DataLoader(dataset=trainer.train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=trainer.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=trainer.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = torch.nn.MSELoss()
    if MODEL_TYPE == 'LINEAR':
        model = LinearPredictor(seq_length=SEQ_LENGTH, hidden_size=HIDDEN_SIZE)
    elif MODEL_TYPE == 'GRU':
        model = GRUPredictor(seq_length=SEQ_LENGTH, num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE)
    elif MODEL_TYPE == 'LSTM':
        model = LSTMPredictor(seq_length=SEQ_LENGTH, num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Start Training...")
    if not os.path.exists(f'ckpts{SEQ_LENGTH}'):
        os.mkdir(f'ckpts{SEQ_LENGTH}')
    if not os.path.exists(f'ckpts{SEQ_LENGTH}/{MODEL_TYPE}'):
        os.mkdir(f'ckpts{SEQ_LENGTH}/{MODEL_TYPE}')    
    train_net(model, train_loader, optimizer, criterion, model_name=MODEL_TYPE, total_iter=ITERATION, ckpt_iter=CKPT_ITER, model_path=f'ckpts{SEQ_LENGTH}/{MODEL_TYPE}')
    print("Start Evaluating...")
    if not os.path.exists(f'pred{SEQ_LENGTH}'):
        os.mkdir(f'pred{SEQ_LENGTH}')
    if not os.path.exists(f'pred{SEQ_LENGTH}/{MODEL_TYPE}'):
        os.mkdir(f'pred{SEQ_LENGTH}/{MODEL_TYPE}')        
    with open(os.path.join(f'pred{SEQ_LENGTH}/{MODEL_TYPE}/summary.txt'), 'w') as f:
        f.write(f'TYPE: {MODEL_TYPE} TRAIN LOSS | VAL LOSS | TEST LOSS\n')
        for i in tqdm(range(0, ITERATION+1, CKPT_ITER)):
            if i == 0:
                continue
            model_path = os.path.join(f'ckpts{SEQ_LENGTH}/{MODEL_TYPE}/', f'{MODEL_TYPE}_iter{i}.pt')
            assert os.path.exists(model_path), model_path
            model.load_state_dict(torch.load(model_path))
            # 只有 shuffle=False 时才可以评估训练集
            loss_train = evaluate_net(model, train_loader, criterion, save_path=f'pred{SEQ_LENGTH}/{MODEL_TYPE}/', save_name=f'train_{MODEL_TYPE}_Iter{i}.csv') 
            loss_val = evaluate_net(model, val_loader, criterion, save_path=f'pred{SEQ_LENGTH}/{MODEL_TYPE}/', save_name=f'val_{MODEL_TYPE}_Iter{i}.csv')
            loss_test = evaluate_net(model, test_loader, criterion, save_path=f'pred{SEQ_LENGTH}/{MODEL_TYPE}/', save_name=f'test_{MODEL_TYPE}_Iter{i}.csv')
            f.write(f'ITER {i}: %.5f | %.5f | %.5f\n' % (loss_train, loss_val, loss_test))
