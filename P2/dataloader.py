import csv
import os
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.autograd import Variable
from tqdm import tqdm

DEVICE = torch.device("cpu")
DEFAULT_TENSOR_TYPE = torch.FloatTensor
torch.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def load_raw(filename='train.csv'):         # 读取csv文件，返回npy数组
    f = open(filename, 'r')
    with f:
        bank_seq_len = None
        banks_npy = None
        reader = csv.reader(f)
        for row in reader:
            bank_data = np.array([float(i) for i in row])  # 735
            if bank_seq_len is not None:
                assert bank_data.shape[0] == bank_seq_len, f"{bank_data.shape[0]}\t{bank_seq_len}"
            else:
                bank_seq_len = bank_data.shape[0]
            if banks_npy is None:
                banks_npy = bank_data
            else:
                banks_npy = np.vstack((banks_npy, bank_data))
    if banks_npy is None:
        raise AssertionError
    return banks_npy


def normalization(train_data, train_label, val_data, val_label, test_data, test_label):  # 对数据MinMaxNorm预处理
    x_train_normer = MinMaxScaler()
    y_train_normer = MinMaxScaler()
    x_val_normer = MinMaxScaler()
    y_val_normer = MinMaxScaler()
    x_test_normer = MinMaxScaler() 
    train_data = x_train_normer.fit_transform(train_data)
    val_data = x_val_normer.fit_transform(val_data)
    test_data = x_test_normer.fit_transform(test_data)
    train_label = y_train_normer.fit_transform(train_label)
    test_label = y_val_normer.fit_transform(test_label)
    val_label = y_val_normer.fit_transform(val_label)
    return train_data, train_label, val_data, val_label, test_data, test_label, y_val_normer


# def get_label(x, seq_length, label_length=56):        # 对csv数据文件滑动窗口划分输入数据和标签
#     """
#         x: np.array('train.csv'), 111x735
#     """
#     assert x.shape[0] == 2 or x.shape[0] == 111, x.shape
#     assert x.shape[1] == 735, x.shape
#     # 滑动窗口获取数据和标签
#     data_np = x[:, :seq_length]
#     label_np =  x[:, seq_length:seq_length+label_length]
#     for idx in range(x.shape[1] - seq_length - label_length):
#         # print(f"{data_np.shape}...")
#         data_np = np.vstack((data_np, x[:, idx:idx+seq_length]))
#         label_np = np.vstack((label_np, x[:, idx+seq_length:idx+seq_length+label_length]))
#     testdata_np = x[:, -seq_length:]
#     return data_np, label_np, testdata_np


# def split(x, y, split_interval=10):                   # 沿时间顺序，每隔split_interval个样本取一个验证集样本
#     train_data = None
#     train_label = None
#     val_data = None
#     val_label = None
#     for i in tqdm(range(1, len(x))):
#         if i % split_interval == 0:
#             if val_data is None and val_label is None:
#                 val_data = x[i]
#                 val_label = y[i]
#             else:
#                 val_data = np.vstack((val_data, x[i]))
#                 val_label = np.vstack((val_label, y[i]))        
#         else:
#             if train_data is None and train_label is None:
#                 train_data = x[i]
#                 train_label = y[i]
#             else:
#                 train_data = np.vstack((train_data, x[i]))
#                 train_label = np.vstack((train_label, y[i]))
#     permutation = np.random.permutation(train_data.shape[0])
#     train_data = train_data[permutation]
#     train_label = train_label[permutation]      
#     return train_data, train_label, val_data, val_label


def split(x, seq_length, label_length=56, ratio=0.7):
    test_x = x[:, -seq_length:]
    threshold_split = int(x[0].shape[-1] * ratio)         # 735 * 0.7 = 514
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    for idx in range(threshold_split):
        if train_x is None:
            train_x = x[:, idx:idx+seq_length]
            train_y = x[:, idx+seq_length:idx+seq_length+label_length]
        else:
            train_x = np.vstack((train_x, x[:, idx:idx+seq_length]))
            train_y = np.vstack((train_y, x[:, idx+seq_length:idx+seq_length+label_length]))
    for idx in range(threshold_split, x[0].shape[-1] - seq_length - label_length):
        if val_x is None:
            val_x = x[:, idx:idx+seq_length]
            val_y = x[:, idx+seq_length:idx+seq_length+label_length]
        else:
            val_x = np.vstack((val_x, x[:, idx:idx+seq_length]))
            val_y = np.vstack((val_y, x[:, idx+seq_length:idx+seq_length+label_length]))
    permutation = np.random.permutation(train_x.shape[0])
    train_x = train_x[permutation]
    train_y = train_y[permutation]
    permutation = np.random.permutation(val_x.shape[0])
    val_x = val_x[permutation]
    val_y = val_y[permutation]
    return train_x, train_y, val_x, val_y, test_x
    

class Trainer():
    def __init__(self, seq_length):
        self.seq_length = seq_length    # 输入长度
        self.normer = None              # MinMaxScalar，用于将输出数据还原到原始数据
        self.train_data = None
        self.train_label = None
        self.val_data = None
        self.val_label = None
        self.test_data = None
        self.test_label = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def collect_data(self, load_npy=False, load_npy_path='./data/'):
        test_label = load_raw('test.csv')
        # 划分训练集，验证集，测试集
        if not load_npy:
            print("Preparing Rawdata...")
            # 获取原始数据
            data = load_raw('train.csv')
            # 打标签，获取训练/验证集的数据和标签，以及测试集数据
            # data, label, test_data = get_label(data, self.seq_length)
            print("Splitting Train/Val/Test data from scratch...")
            # train_data, train_label, val_data, val_label = split(data, label)
            train_data, train_label, val_data, val_label, test_data = split(data, self.seq_length)
            if not os.path.exists(load_npy_path):
                os.mkdir(load_npy_path)
            np.save(os.path.join(load_npy_path, 'train_x.npy'), train_data)
            np.save(os.path.join(load_npy_path, 'train_y.npy'), train_label)
            np.save(os.path.join(load_npy_path, 'val_x.npy'), val_data)
            np.save(os.path.join(load_npy_path, 'val_y.npy'), val_label)
            np.save(os.path.join(load_npy_path, 'test_x.npy'), test_data)
        else:
            print(f"Loading Train/Val/Test data from {load_npy_path}...")
            assert os.path.exists(load_npy_path)
            assert os.path.exists(os.path.join(load_npy_path, 'train_x.npy'))
            assert os.path.exists(os.path.join(load_npy_path, 'train_y.npy'))  # 保存下来用来计算sMAPE
            assert os.path.exists(os.path.join(load_npy_path, 'val_x.npy'))
            assert os.path.exists(os.path.join(load_npy_path, 'val_y.npy'))    # 保存下来用来计算sMAPE
            assert os.path.exists(os.path.join(load_npy_path, 'test_x.npy'))
            train_data = np.load(os.path.join(load_npy_path, 'train_x.npy'))
            train_label = np.load(os.path.join(load_npy_path, 'train_y.npy'))
            val_data = np.load(os.path.join(load_npy_path, 'val_x.npy'))
            val_label = np.load(os.path.join(load_npy_path, 'val_y.npy'))
            test_data = np.load(os.path.join(load_npy_path, 'test_x.npy'))
            try:
                assert train_data.shape[1] == self.seq_length
                assert val_data.shape[1] == self.seq_length
                assert test_data.shape[1] == self.seq_length
            except AssertionError:
                raise ValueError('npy shape is not consistent with self.seq_length.')        
        # 正则化
        print(f"Trainset Shape: {train_data.shape}; Valset Shape: {val_data.shape}; Testset Shape: {test_data.shape}")         
        train_data, train_label, val_data, val_label, test_data, test_label, normer = normalization(train_data, train_label, val_data, val_label, test_data, test_label)
        self.normer = normer
        # 生成 Dataset & Dataloader
        self.train_data = Variable(torch.Tensor(train_data))
        self.train_label = Variable(torch.Tensor(train_label))
        self.val_data = Variable(torch.Tensor(val_data))
        self.val_label = Variable(torch.Tensor(val_label))
        self.test_data = Variable(torch.Tensor(test_data))
        self.test_label = Variable(torch.Tensor(test_label))
        self.train_dataset = torch.utils.data.TensorDataset(self.train_data, self.train_label)
        self.val_dataset = torch.utils.data.TensorDataset(self.val_data, self.val_label)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_data, self.test_label)
        # 生成 numpy 测试标签
        self.test_label = test_label


if __name__ == '__main__':
    SEQ_LENGTH = 56
    NPY_PATH = f"./data{SEQ_LENGTH}/"
    trainer = Trainer(seq_length=56)
    trainer.collect_data(load_npy=False, load_npy_path=NPY_PATH)
    trainer = Trainer(seq_length=56)
    trainer.collect_data(load_npy=True, load_npy_path=NPY_PATH)
