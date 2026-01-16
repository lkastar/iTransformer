import torch
from torch import nn

class TrendFlow(nn.Module):
    """
    用于时间序列趋势预测的 MLP 网络。
    采用 Channel Independence 策略：所有通道共享同一套 MLP 权重。
    
    Structure:
        Input -> Linear -> GELU -> Dropout -> LayerNorm
              -> Linear -> GELU -> Dropout -> LayerNorm
              -> Linear -> Output
    """
    def __init__(self, seq_len: int, pred_len: int, hidden_ratio: int = 4, dropout: float = 0.1):
        """
        Args:
            seq_len (int): 输入序列长度
            pred_len (int): 预测序列长度
            hidden_ratio (int): 隐藏层相对于 pred_len 的倍率，默认 4
            dropout (float): Dropout 概率，默认 0.1
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        hidden_dim = pred_len * hidden_ratio
        
        # Layer 1: 输入投影 + 升维
        self.layer1 = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Layer 2: 隐藏层
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Layer 3: 输出投影
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Length, Channel] (Standard Time Series Format)
        Returns:
            out: [Batch, Pred_Length, Channel]
        """
        B, L, C = x.shape
        
        # Channel Independence: Flatten Batch and Channel
        # [Batch, Length, Channel] -> [Batch * Channel, Length]
        x = x.permute(0, 2, 1).reshape(B * C, L)

        # MLP Forward
        x = self.layer1(x)  # [B*C, hidden_dim]
        x = self.layer2(x)  # [B*C, hidden_dim]
        x = self.head(x)    # [B*C, pred_len]

        # Reshape back
        # [Batch * Channel, Pred_Length] -> [Batch, Channel, Pred_Length]
        x = x.reshape(B, C, self.pred_len)
        
        # Permute to [Batch, Pred_Length, Channel]
        x = x.permute(0, 2, 1)

        return x


class TrendFlowRepr(nn.Module):
    """
    Trend representation network (Channel Independence).
    Input : [B, L, C]
    Output: [B, C, D] where D=d_model
    """
    def __init__(self, seq_len: int, d_model: int, hidden_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        hidden_dim = d_model * hidden_ratio

        self.layer1 = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: [B, C, D]
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x)                          # [B*C, D]
        x = x.reshape(B, C, self.d_model)         # [B, C, D]
        return x


# non-linear patch style (from xPatch)
# class TrendFlow(nn.Module):
#     """
#     用于时间序列趋势预测的 MLP 网络。
#     采用 Channel Independence 策略：所有通道共享同一套 MLP 权重。
    
#     Structure:
#         Input -> [Linear -> Permute -> AvgPool -> Permute -> Norm -> Activation] x 2 -> Linear -> Output
#     """
#     def __init__(self, seq_len: int, pred_len: int, hidden_ratio: int = 4):
#         """
#         Args:
#             seq_len (int): 输入序列长度
#             pred_len (int): 预测序列长度
#             hidden_ratio (int): 第一层升维倍率，默认 4
#         """
#         super().__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
        
#         # 维度计算
#         dim_l1_out = pred_len * hidden_ratio
#         dim_l1_pool = dim_l1_out // 2  # Kernel=2, Stride=2 implies /2
        
#         dim_l2_in = dim_l1_pool
#         dim_l2_out = pred_len
#         dim_l2_pool = dim_l2_out // 2
        
#         # Layer 1: 升维 + 下采样
#         self.layer1 = nn.Sequential(
#             nn.Linear(seq_len, dim_l1_out),
#             nn.GELU(), # 增加非线性
#             # 注意：AvgPool 需要 (N, C, L) 格式，我们会在 forward 中处理维度，
#             # 或者使用 Linear 层的输出作为 Feature，这里用 AvgPool 是为了平滑/压缩特征
#         )
#         self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
#         self.ln1 = nn.LayerNorm(dim_l1_pool)

#         # Layer 2: 降维 + 下采样
#         self.layer2 = nn.Sequential(
#             nn.Linear(dim_l2_in, dim_l2_out),
#             nn.GELU()
#         )
#         self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
#         self.ln2 = nn.LayerNorm(dim_l2_pool)

#         # Layer 3: 最终投影
#         self.head = nn.Linear(dim_l2_pool, pred_len)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: [Batch, Length, Channel] (Standard Time Series Format)
#         Returns:
#             out: [Batch, Pred_Length, Channel]
#         """
#         # x: [Batch, Input_Len, Channel]
#         B, I, C = x.shape
        
#         # 1. Channel Independence: Flatten Batch and Channel
#         # 变换为 [Batch * Channel, Input_Len]
#         x = x.permute(0, 2, 1).reshape(B * C, I)

#         # === Block 1 ===
#         x = self.layer1(x)              # [B*C, pred_len*4]
        
#         # AvgPool 需要 [N, C, L] 格式。
#         # 这里的 "L" 是特征维度。我们需要把特征维度当作 Time 轴来 Pool，或者当作 C 轴？
#         # 原意图是对 Linear 输出的 Feature 向量进行降维压缩。
#         # Linear 输出 Shape: [B*C, Features]
#         # 为了使用 AvgPool1d 压缩 Features，必须将其变为 [B*C, 1, Features]
#         x = x.unsqueeze(1)              # [B*C, 1, pred_len*4]
#         x = self.pool1(x)               # [B*C, 1, pred_len*2]
#         x = x.squeeze(1)                # [B*C, pred_len*2]
#         x = self.ln1(x)

#         # === Block 2 ===
#         x = self.layer2(x)              # [B*C, pred_len]
        
#         x = x.unsqueeze(1)              # [B*C, 1, pred_len]
#         x = self.pool2(x)               # [B*C, 1, pred_len//2]
#         x = x.squeeze(1)                # [B*C, pred_len//2]
#         x = self.ln2(x)

#         # === Head ===
#         x = self.head(x)                # [B*C, pred_len]

#         # 2. Reshape back
#         # [Batch * Channel, Output_Len] -> [Batch, Channel, Output_Len]
#         x = x.reshape(B, C, self.pred_len)
        
#         # 3. Permute to [Batch, Output_Len, Channel]
#         x = x.permute(0, 2, 1)

#         return x