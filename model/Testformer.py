import torch
from torch import nn

from layers.ema import Decomp
from layers.StandardNorm import Normalize
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

from layers.PAEmbedv2 import PAEmbedding
from layers.mul import MultiResSeasonalEmbedding


class Model(nn.Module):
    
    def __init__(self, config, alpha=0.3):
        super(Model, self).__init__()
        
        self.config = config
        self.decomp = Decomp(alpha)
        self.normlizer = Normalize(config.enc_in, affine=True)
        self.trend_net = nn.Linear(config.seq_len, config.d_model)
        self.season_net = SeasonFlow(config)
        self.projector = nn.Linear(config.d_model, config.pred_len)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc = self.normlizer(x_enc, mode='norm')
        
        seasonal_init, trend_init = self.decomp(x_enc)
        # [B, L, C] -> [B, C, L]
        # seasonal_init暂时适配iTransformer的输入格式
        trend_init = trend_init.permute(0, 2, 1)  # [B, C, L]
        
        x_trend = self.trend_net(trend_init) # [B, L, C]
        x_seasonal = self.season_net(seasonal_init) # [B, L, C]
        
        out = self.projector(x_seasonal + x_trend).permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        
        out = self.normlizer(out, mode='denorm')
        
        return out
    

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
        
        # self.pa_embedding = PAEmbedding(configs.d_model)
        
        self.mul_embedding = MultiResSeasonalEmbedding(
            d_model=configs.d_model,
            target_lengths=[configs.seq_len, configs.seq_len // 2, configs.seq_len // 4],
            anchor_len=configs.seq_len // 2
        )
        
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
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

    def forecast(self, x_enc):
    
        _, N, _ = x_enc.shape # B, N, L
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # NOTE 简单的时间戳特征影响模型性能
        # NOTE iTransformer的invert操作在Embedding阶段完成
        # NOTE [B, L, N] -> [B, N, D] 在变量维度上进行Embedding
        # enc_out = self.enc_embedding(x=x_enc, x_mark=None) # covariates (e.g timestamp) can be also embedded as tokens
        # x_enc = self.pa_embedding(x_enc)
        enc_out = self.mul_embedding(x_enc)  # [B, N, D]
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        return enc_out, attns
    
    def forward(self, x_enc):
        """
        iTransformer as encoder
        """
        enc_out, attns = self.forecast(x_enc)

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, N, E]