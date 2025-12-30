import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    

def create_plot_figure(gt, pd, title_suffix=""):
    """
    创建一个 matplotlib figure 对象，不保存到磁盘。
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(gt, label='GroundTruth', linewidth=1.5)
    plt.plot(pd, label='Prediction', linewidth=1.5, linestyle='--')
    plt.legend()
    plt.title(f"Forecast Result {title_suffix}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def plot_series(x, y, label_len, 
                      label_x='History (x)', label_y='Prediction (y)',
                      color_x='#1f77b4', color_y='#ff7f0e',
                      title='Sequence Plot with Overlap'):
    """
    绘制两段序列，其中 y 的前 label_len 长度与 x 的末尾重叠。
    
    参数:
    x : array-like, 历史序列 (Context)
    y : array-like, 包含重叠部分的预测序列 (Decoder Output)
    label_len : int, 重叠的长度 (y 的前 label_len 个点对应 x 的最后 label_len 个点)
    """
    x = np.array(x)
    y = np.array(y)
    
    # 1. 构造横坐标
    # x 的索引: 0, 1, ..., len(x)-1
    indices_x = np.arange(0, len(x))
    
    # y 的索引: 从 (len(x) - label_len) 开始
    # 这样 y 的第一个点就会对齐到 x 的倒数第 label_len 个点的位置
    start_y = len(x) - label_len
    indices_y = np.arange(start_y, start_y + len(y))
    
    # 2. 绘图
    plt.figure(figsize=(7, 3))
    
    # 绘制 x (底层)
    plt.plot(indices_x, x, color=color_x, label=label_x, linewidth=2)
    
    # 绘制 y (上层，通常希望看到预测值覆盖在历史值上)
    # 使用 alpha 设置一点透明度，这样可以看到重叠部分的差异
    plt.plot(indices_y, y, color=color_y, label=label_y, linewidth=2, alpha=0.8)
    
    # 3. 添加辅助线 (可选)
    # 在重叠开始的地方画一条竖线，标记 "Decoder Start"
    plt.axvline(x=start_y, color='gray', linestyle='--', alpha=0.6, label='Overlap Start')
    
    # 在重叠结束(真正预测开始)的地方画一条竖线
    if label_len < len(y):
        pred_start = len(x)
        plt.axvline(x=pred_start, color='red', linestyle=':', alpha=0.6, label='Future Start')

    plt.title(title)
    plt.xlabel('Time Step / Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
