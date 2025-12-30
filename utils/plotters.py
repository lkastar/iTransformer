import matplotlib.pyplot as plt
import numpy as np

# plt.switch_backend('TkAgg')

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