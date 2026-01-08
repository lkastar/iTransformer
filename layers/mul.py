import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 基础工具函数
# ==========================================

def multi_scale_resampling(x, target_lengths):
    """
    对输入时间序列进行多尺度重采样 (Pyramid Resampling)。
    Args:
        x: [Batch, Variate, Time]
        target_lengths: List[int], 例如 [96, 48, 24]
    Returns:
        resampled_dict: {length: tensor [Batch, Variate, length]}
    """
    # 自动修正维度: [Variate, Time] -> [1, Variate, Time]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    
    resampled_dict = {}
    
    for length in target_lengths:
        if length == x.shape[-1]:
            resampled_dict[length] = x
        else:
            # F.interpolate 对最后一大维 (Time) 进行线性插值
            # mode='linear' 要求输入为 3D: [Batch, Channel, Time]
            x_resized = F.interpolate(x, size=length, mode='linear', align_corners=False)
            resampled_dict[length] = x_resized
            
    return resampled_dict

def resample_weight(old_weight: torch.Tensor, new_len: int):
    """
    Flex-Resize 核心: 将权重矩阵从 [D, Old] 变形为 [D, New]
    利用伪逆矩阵保持特征映射的物理意义不变。
    """
    # old_weight: [d_model, anchor_len]
    assert old_weight.dim() == 2
    old_len = old_weight.size(1)
    
    if old_len == new_len:
        return old_weight
    
    # 转置为 [Old, D] 以便计算
    old = old_weight.T 
    factor = new_len / old_len
    
    # 构造变换基
    basis = torch.eye(old_len, dtype=torch.float32, device=old.device)
    # 插值: [1, 1, Old] -> [1, 1, New]
    resize_mat = F.interpolate(basis.unsqueeze(0).unsqueeze(0), 
                               size=(new_len, old_len), 
                               mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    
    # 伪逆求解: [New, Old]
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)
    
    # 映射回新权重
    resampled = resize_mat_pinv @ old * math.sqrt(factor)
    
    return resampled.T # [d_model, new_len]

# ==========================================
# 2. 多尺度弹性 Embedding 模块
# ==========================================

class MultiResSeasonalEmbedding(nn.Module):
    def __init__(self, d_model, target_lengths=[96, 48, 24], anchor_len=48, dropout=0.1):
        """
        Args:
            d_model: 输出维度
            target_lengths: 需要重采样的尺度列表 (建议包含原始长度)
            anchor_len: 基准权重的长度 (建议选中间值，如 48)
        """
        super().__init__()
        self.target_lengths = target_lengths
        self.d_model = d_model
        
        # --- 核心组件 1: 共享锚点权重 ---
        # 我们只维护这一组参数，其他尺度的权重都由它动态变形而来
        # 物理含义：这是“标准尺度”下的波形特征提取器
        self.anchor_weight = nn.Parameter(torch.empty(d_model, anchor_len))
        self.anchor_bias = nn.Parameter(torch.zeros(d_model)) 
        
        # 初始化
        nn.init.kaiming_uniform_(self.anchor_weight, a=math.sqrt(5))
        
        # --- 核心组件 2: 多尺度特征融合 ---
        # 输入维度 = d_model * 尺度数量 (Concat)
        # 输出维度 = d_model (Fusion)
        self.fusion_layer = nn.Sequential(
            nn.Linear(len(target_lengths) * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 缓存动态生成的权重，避免重复计算
        self._weight_cache = {}

    def _get_projector(self, length):
        """获取适配指定长度的权重矩阵 [D, Length]"""
        if length in self._weight_cache:
            return self._weight_cache[length]
        
        # 动态变形: Anchor(48) -> New(Length)
        w = resample_weight(self.anchor_weight, length)
        self._weight_cache[length] = w
        return w

    def forward(self, x):
        """
        Input: [Batch, Variate, Time] (通常是 Decomp 后的 Seasonal 成分)
        Output: [Batch, Variate, d_model]
        """
        x = x.permute(0, 2, 1)  # [B, N, L]
        
        self._weight_cache = {} # 清空缓存 (若 target_lengths 固定，其实可以不以 batch 为单位清空，但在训练中为了梯度安全通常清空)
        
        # 1. 多尺度重采样 (Pyramid Resampling)
        # 得到字典: {96: Tensor[B,N,96], 48: Tensor[B,N,48], ...}
        resampled_dict = multi_scale_resampling(x, self.target_lengths)
        
        tokens_list = []
        
        # 2. 弹性投影 (Elastic Projection)
        for length in self.target_lengths:
            x_res = resampled_dict[length] # [B, N, L]
            
            # 获取权重 [D, L]
            w_L = self._get_projector(length)
            
            # 线性映射: x @ w.T + b
            # [B, N, L] @ [L, D] -> [B, N, D]
            # 这一步将不同长度的时间序列，映射到了统一的语义空间 D
            out = F.linear(x_res, w_L, self.anchor_bias)
            
            tokens_list.append(out)
            
        # 3. 融合 (Fusion)
        # 将不同分辨率的特征拼接
        concat_tokens = torch.cat(tokens_list, dim=-1) # [B, N, 3*D]
        
        # 融合为一个 Variate Token
        final_token = self.fusion_layer(concat_tokens) # [B, N, D]
        
        return final_token