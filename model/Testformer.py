import torch
from torch import nn
from einops import rearrange
from layers.ema import Decomp
from layers.StandardNorm import Normalize
from layers.TrendFlow import TrendFlow
from layers.SeasonFlow import SeasonFlow
from layers.masked_attention import Mahalanobis_mask



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
        
        x_trend = self.trend_net(trend_init) # [B, L, C]
        x_seasonal = self.season_net(seasonal_init, x_mark_enc, mask=mah_mask) # [B, L, C]
        
        out = self.projector(x_seasonal + rearrange(x_trend, 'b d c -> b c d')).permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        
        out = self.normlizer(out, mode='denorm')
        
        return out