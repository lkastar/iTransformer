import torch
from torch import nn
from einops import rearrange
from layers.ema import Decomp
from layers.StandardNorm import Normalize
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.masked_attention import FullAttention, Mahalanobis_mask

from layers.PAEmbedv3 import PAEmbedding
from layers.mul import MultiResSeasonalEmbedding


class Model(nn.Module):
    
    def __init__(self, config, alpha=0.3):
        super(Model, self).__init__()
        
        self.config = config
        self.decomp = Decomp(alpha)
        self.normlizer = Normalize(config.enc_in, affine=True)
        # self.trend_net = nn.Linear(config.seq_len, config.d_model)
        self.trend_net = TrendFlow(config.seq_len, config.d_model)
        self.season_net = SeasonFlow(config)
        # self.projector = nn.Linear(config.d_model, config.pred_len)
        self.mask_generator = Mahalanobis_mask(config.seq_len)

        self.projector = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len * 4),
            nn.LayerNorm(config.pred_len * 4),
            nn.GELU(),

            nn.Linear(config.pred_len * 4, config.pred_len * 2),
            nn.LayerNorm(config.pred_len * 2),
            nn.GELU(),

            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.LayerNorm(config.pred_len),

            nn.Linear(config.pred_len, config.pred_len),
        )
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, L, C]
        x_enc = self.normlizer(x_enc, mode='norm')
        mah_mask = self.mask_generator(rearrange(x_enc, 'b l c -> b c l'))
        
        seasonal_init, trend_init = self.decomp(x_enc)
        # [B, L, C] -> [B, C, L]
        # trend_init = rearrange(trend_init, 'b l c -> b c l')  # [B, C, L]
        # seasonal兼容iTransformer结构
        # seasonal_init = seasonal_init
        
        x_trend = self.trend_net(trend_init) # [B, L, C]
        x_seasonal = self.season_net(seasonal_init, mask=mah_mask) # [B, L, C]
        
        out = self.projector(x_seasonal + rearrange(x_trend, 'b d c -> b c d')).permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        
        out = self.normlizer(out, mode='denorm')
        
        return out

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

class SeasonFlow(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(SeasonFlow, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.pa_embedding = PAEmbedding(configs.d_model)
        
        # self.mul_embedding = MultiResSeasonalEmbedding(
        #     d_model=configs.d_model,
        #     target_lengths=[configs.seq_len, configs.seq_len // 2, configs.seq_len // 4],
        #     anchor_len=configs.seq_len // 2
        # )
        
        # mask from https://github.com/decisionintelligence/DUET
        self.mask_generator = Mahalanobis_mask(configs.seq_len)
        
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, mask=None):
        """
        Docstring for forecast
        
        :param x_enc: B, L, C
        """
        _, N, _ = x_enc.shape # B, N, L
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # NOTE 简单的时间戳特征影响模型性能
        # NOTE iTransformer的invert操作在Embedding阶段完成
        # NOTE [B, L, N] -> [B, N, D] 在变量维度上进行Embedding
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        # enc_out = self.pa_embedding(x_enc)
        # x_enc = self.mul_embedding(x_enc)  # [B, N, D]

        # Generate Mahalanobis mask
        # mask = self.mask_generator(rearrange(x_enc, 'b l c -> b c l'))
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=mask)

        return enc_out, attns
    
    def forward(self, x_enc, mask=None):
        """
        iTransformer as encoder
        """
        enc_out, attns = self.forecast(x_enc, mask=mask)
        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, N, E]