import torch
from torch import nn
from einops import rearrange

from layers.Embed import DataEmbedding_inverted, DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.masked_attention import FullAttention, Mahalanobis_mask
from layers.mul import MultiResSeasonalEmbedding

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
        self.data_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout
        )
        
        self.mul_embedding = MultiResSeasonalEmbedding(
            d_model=configs.d_model,
            target_lengths=[configs.seq_len, configs.seq_len // 2, configs.seq_len // 4],
            anchor_len=configs.seq_len // 2
        )
        
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

    def forecast(self, x_enc, x_mark_enc=None, mask=None):
        """
        Docstring for forecast
        
        :param x_enc: B, L, C
        """
        _, N, _ = x_enc.shape # B, N, L
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L C -> B C E                (B L C -> B L E in the vanilla Transformer)
        # NOTE 简单的时间戳特征影响模型性能
        # NOTE iTransformer的invert操作在Embedding阶段完成
        # NOTE [B, L, C] -> [B, C, L] -> [B, C, E] 在变量维度上进行Embedding
        # enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        # enc_out = self.pa_embedding(x_enc)
        enc_out = self.mul_embedding(rearrange(x_enc, 'b l c -> b c l'))  # [B, C, E]
        # enc_out = rearrange(enc_out, 'b c e -> b c e')

        # enc_out = self.data_embedding(rearrange(x_enc, 'b l n -> b n l'), x_mark_enc)

        # Generate Mahalanobis mask
        # mask = self.mask_generator(rearrange(x_enc, 'b l c -> b c l'))
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=mask)

        return enc_out, attns
    
    def forward(self, x_enc, x_mark_enc=None, mask=None):
        """
        iTransformer as encoder
        """
        enc_out, attns = self.forecast(x_enc, x_mark_enc, mask=mask)
        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, N, E]