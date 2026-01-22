import torch
from torch import nn

from layers.ema import Decomp


class EMADecomposedLoss(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        trend_weight=1.0,
        seasonal_weight=1.0,
        base_loss=None,
    ):
        super().__init__()
        self.decomp = Decomp(alpha)
        self.trend_weight = float(trend_weight)
        self.seasonal_weight = float(seasonal_weight)
        self.base_loss = base_loss if base_loss is not None else nn.MSELoss()

    def forward(self, y_hat, y):
        seasonal_hat, trend_hat = self.decomp(y_hat)
        seasonal, trend = self.decomp(y)

        loss_trend = self.base_loss(trend_hat, trend)
        loss_seasonal = self.base_loss(seasonal_hat, seasonal)

        return self.trend_weight * loss_trend + self.seasonal_weight * loss_seasonal


class DBLoss(nn.Module):
    """自定义分解损失函数（趋势+季节双损失）"""

    def __init__(self, alpha, beta):
        super().__init__()
        self.decomp = Decomp(alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, pred, target):
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss

