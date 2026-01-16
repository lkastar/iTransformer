import torch
import torch.nn as nn

class FiLMFusion_BCD(nn.Module):
    """
    h = (1 + scale*gamma(h_t)) ⊙ h_s + scale*beta(h_t) + h_t
    Inputs: h_s, h_t: [B, C, D]
    Output: h: [B, C, D]
    """
    def __init__(self, d_model: int, hidden: int = 256, init_scale: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # init_scale=0 -> 退化为 h_s + h_t，训练很稳
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(h_t)  # [B,C,D]
        beta  = self.beta_net(h_t)   # [B,C,D]
        h_s_mod = (1.0 + self.scale * gamma) * h_s + self.scale * beta
        h = h_s_mod + h_t
        return self.drop(h)


class DecompInteractionProjector(nn.Module):
    """
    Inputs:
      h_s, h_t: [B, C, D]
    Output:
      y: [B, C, P]
    """
    def __init__(self, d_model: int, pred_len: int, film_init_scale: float = 0.0, film_hidden: int = 256, film_dropout: float = 0.0):
        super().__init__()
        self.fusion = FiLMFusion_BCD(d_model=d_model, hidden=film_hidden, init_scale=film_init_scale, dropout=film_dropout)

        # 你的 projector：D -> pred_len
        self.projector = nn.Sequential(
            nn.Linear(d_model, pred_len * 4),
            nn.LayerNorm(pred_len * 4),
            nn.GELU(),

            nn.Linear(pred_len * 4, pred_len * 2),
            nn.LayerNorm(pred_len * 2),
            nn.GELU(),

            nn.Linear(pred_len * 2, pred_len),
            nn.LayerNorm(pred_len),

            nn.Linear(pred_len, pred_len),
        )

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        h = self.fusion(h_s, h_t)      # [B,C,D]
        y = self.projector(h)          # [B,C,P]  (Linear/LayerNorm 都作用在最后一维)
        return y
