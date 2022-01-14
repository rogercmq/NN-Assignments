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

from models import LinearPredictor, LSTMPredictor, GRUPredictor, Seq2Seq
from dataloader import Trainer, load_raw
from metric import smape

def train_net(model, dataloader, optimizer, criterion, total_iter=10000, log_iter=1000, ckpt_iter=1000, model_name='model', model_path='./ckpts'):
    cur_iter = 0
    for _ in range(10000):
        for _, (batch_x, batch_y) in enumerate (dataloader): 
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
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
def evaluate_net(model, dataloader, criterion, save_path, save_name='result.csv', normer=None):
    results = None
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    total_iters = 0
    total_loss = 0
    for _, (batch_x, batch_y) in enumerate (dataloader): 
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        outputs = model(batch_x)
        total_loss += criterion(outputs, batch_y)
        total_iters += 1
        pred = outputs.detach().cpu().numpy()
        if normer is not None:
            pred = normer.inverse_transform(pred)
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


def cal_smape(pred_filename, mode, gt_path):
    preds = load_raw(pred_filename)
    if mode == 'train':
        labels = np.load(os.path.join(gt_path, 'train_y.npy'))
    elif mode == 'val':
        labels = np.load(os.path.join(gt_path, 'val_y.npy'))
    else:
        labels = load_raw('test.csv')
    assert preds.shape == labels.shape, f'{preds.shape} | {labels.shape}'
    avg_smape = 0.
    for i in range(len(preds)):
        avg_smape += smape(preds[i], labels[i])
    return avg_smape / len(preds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=56)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='LINEAR')
    args = parser.parse_args()
    
    SEQ_LENGTH = args.seq_length
    HIDDEN_SIZE = args.hidden_size
    MODEL_TYPE = args.model_type # GRU | LSTM | LINEAR | SEQ2SEQ
    
    INPUT_SIZE = 1
    NUM_LAYERS = 1
    LR = 0.001
    BATCH_SIZE = 64
    ITERATION = 120000
    CKPT_ITER = 20000
    
    trainer = Trainer(seq_length=SEQ_LENGTH)
    if os.path.exists(f'data{SEQ_LENGTH}'):
        trainer.collect_data(load_npy=True, load_npy_path=f'data{SEQ_LENGTH}')
    else:
        trainer.collect_data(load_npy=False, load_npy_path=f'data{SEQ_LENGTH}')
    train_loader = torch.utils.data.DataLoader(dataset=trainer.train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=trainer.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=trainer.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = torch.nn.MSELoss() # MSELoss | L1Loss
    if MODEL_TYPE == 'LINEAR':
        model = LinearPredictor(seq_length=SEQ_LENGTH, hidden_size=HIDDEN_SIZE)
    elif MODEL_TYPE == 'GRU':
        model = GRUPredictor(seq_length=SEQ_LENGTH, num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE)
    elif MODEL_TYPE == 'LSTM':
        model = LSTMPredictor(seq_length=SEQ_LENGTH, num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE)
    elif MODEL_TYPE == 'SEQ2SEQ':
        model = Seq2Seq()
    model = model.cuda()
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
        f.write(f'TYPE: {MODEL_TYPE} TRAIN LOSS | VAL LOSS | TEST LOSS | TRAIN SMAPE | VAL SMAPE | TEST SMAPE \n')
        for i in tqdm(range(0, ITERATION+1, CKPT_ITER)):
            if i == 0:
                continue
            model_path = os.path.join(f'ckpts{SEQ_LENGTH}/{MODEL_TYPE}/', f'{MODEL_TYPE}_iter{i}.pt')
            assert os.path.exists(model_path), model_path
            model.load_state_dict(torch.load(model_path))
            # 只有 shuffle=False 时才可以评估训练集
            normer = trainer.normer
            loss_train = evaluate_net(model, train_loader, criterion, save_path=f'pred{SEQ_LENGTH}/{MODEL_TYPE}/', save_name=f'train_{MODEL_TYPE}_Iter{i}.csv', normer=normer) 
            loss_val = evaluate_net(model, val_loader, criterion, save_path=f'pred{SEQ_LENGTH}/{MODEL_TYPE}/', save_name=f'val_{MODEL_TYPE}_Iter{i}.csv', normer=normer)
            loss_test = evaluate_net(model, test_loader, criterion, save_path=f'pred{SEQ_LENGTH}/{MODEL_TYPE}/', save_name=f'test_{MODEL_TYPE}_Iter{i}.csv', normer=normer)
            smape_train = cal_smape(f'pred{SEQ_LENGTH}/{MODEL_TYPE}/train_{MODEL_TYPE}_Iter{i}.csv', 'train', gt_path=f'data{SEQ_LENGTH}')
            smape_val = cal_smape(f'pred{SEQ_LENGTH}/{MODEL_TYPE}/val_{MODEL_TYPE}_Iter{i}.csv', 'val', gt_path=f'data{SEQ_LENGTH}')
            smape_test = cal_smape(f'pred{SEQ_LENGTH}/{MODEL_TYPE}/test_{MODEL_TYPE}_Iter{i}.csv', 'test', gt_path=f'data{SEQ_LENGTH}')
            f.write(f'ITER {i}: %.5f | %.5f | %.5f | %.5f | %.5f | %.5f\n' % (loss_train, loss_val, loss_test, smape_train, smape_val, smape_test))
