"""
ChannelPeriodFlexEnhancer: 通道独立周期增强模块

将时间序列从 [B, L, C] 转换为 [B, C, D]，用于 iTransformer 的变量 token attention。

模块职责：
- 通道独立增强：增强 z_c 仅依赖 x[:,:,c]
- 不做跨通道计算（不 attention，不 conv across C）
- 输出 token 直接喂 iTransformer：Z[B,C,D]

Reference: PLAN.md
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- utils: patchify ----------
def patchify_1d(x_c: torch.Tensor, p: int) -> torch.Tensor:
    """
    将单通道时间序列切分为 patches。
    
    Args:
        x_c: [B, L] 单通道时间序列
        p: 周期/patch 长度
        
    Returns:
        [B, N, p], where N = floor(L/p)
        fallback: if N==0 -> interpolate to length p and return N=1
    """
    B, L = x_c.shape
    p = int(p)
    if p <= 0:
        raise ValueError(f"period p must be >0, got {p}")

    N = L // p
    if N <= 0:
        # interpolate to length p -> one patch
        x_rs = F.interpolate(x_c.unsqueeze(1), size=p, mode="linear", align_corners=False).squeeze(1)
        return x_rs.unsqueeze(1)  # [B,1,p]

    x_trim = x_c[:, :N * p]          # [B, N*p]
    return x_trim.view(B, N, p)      # [B, N, p]


# ---------- utils: token resize along N ----------
def resize_tokens(tokens: torch.Tensor, n_fix: int) -> torch.Tensor:
    """
    对齐 token 序列长度。
    
    Args:
        tokens: [B, N, D] token 序列
        n_fix: 目标 token 数量
        
    Returns:
        [B, n_fix, D]
    """
    B, N, D = tokens.shape
    if N == n_fix:
        return tokens
    x = tokens.transpose(1, 2)  # [B, D, N]
    x = F.interpolate(x, size=n_fix, mode="linear", align_corners=False)
    return x.transpose(1, 2)    # [B, n_fix, D]


# ---------- utils: ACF at lag ----------
def acf_at_lag(x_c: torch.Tensor, p: int, eps: float = 1e-6) -> torch.Tensor:
    """
    计算指定 lag 的自相关系数。
    
    Args:
        x_c: [B, L] 单通道时间序列
        p: lag 值（周期）
        eps: 数值稳定性常数
        
    Returns:
        [B] 每个样本的 ACF@p
    """
    B, L = x_c.shape
    p = int(p)
    if p <= 0 or p >= L:
        return x_c.new_zeros(B)

    a = x_c[:, :-p]
    b = x_c[:, p:]
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).mean(dim=1)
    den = (a.pow(2).mean(dim=1).sqrt() * b.pow(2).mean(dim=1).sqrt()).clamp_min(eps)
    return num / den


# ---------- FlexResize: resample patch embedding weights ----------
@torch.no_grad()
def resample_patchemb(old: torch.Tensor, new_patch_len: int) -> torch.Tensor:
    """
    重采样 patch embedding 权重（LightGTS 风格）。
    
    使用 pseudo-inverse 和 sqrt(length_factor) 进行权重重采样。
    
    Args:
        old: [D, P_ref] 原始权重
        new_patch_len: 目标 patch 长度
        
    Returns:
        [D, new_patch_len] 重采样后的权重
    """
    assert old.dim() == 2, "old must be 2D [D, P_ref]"
    D, P_ref = old.shape
    new_patch_len = int(new_patch_len)
    if P_ref == new_patch_len:
        return old

    old_t = old.T  # [P_ref, D]
    factor = new_patch_len / P_ref

    # resize matrix via interpolating identity
    basis = torch.eye(P_ref, dtype=old.dtype, device=old.device)  # [P_ref, P_ref]
    # treat basis as [1, C=P_ref, L=P_ref] and resize L -> new_patch_len
    resize_mat = F.interpolate(basis.unsqueeze(0), size=new_patch_len, mode="linear", align_corners=False).squeeze(0)
    resize_mat = resize_mat.T  # [new, P_ref]

    pinv = torch.linalg.pinv(resize_mat.T)       # pseudo-inverse
    resampled = pinv @ old_t * math.sqrt(factor) # [new, D]
    return resampled.T                           # [D, new]


class FlexPatchEmbedding(nn.Module):
    """
    弹性 Patch Embedding 模块。
    
    基于母核 Linear(P_ref -> D)，对任意 patch 长度 p 进行权重重采样。
    使用缓存避免重复计算 pinv。
    
    Args:
        d_model: 输出维度
        p_ref: 参考 patch 长度（母核大小）
    """
    def __init__(self, d_model: int, p_ref: int):
        super().__init__()
        self.d_model = int(d_model)
        self.p_ref = int(p_ref)
        self.base = nn.Linear(self.p_ref, self.d_model)
        self._w_cache: Dict[int, torch.Tensor] = {}  # p -> [D,p]

    def get_weight(self, p: int) -> torch.Tensor:
        """获取指定 patch 长度的权重（带缓存）。"""
        p = int(p)
        w = self._w_cache.get(p, None)
        # 检查缓存有效性：device 和 dtype 必须匹配
        if w is None or (w.device != self.base.weight.device) or (w.dtype != self.base.weight.dtype):
            w = resample_patchemb(self.base.weight.data, p)
            self._w_cache[p] = w
        return w

    def clear_cache(self):
        """手动清理权重缓存。"""
        self._w_cache.clear()

    def forward(self, patches: torch.Tensor, p: int) -> torch.Tensor:
        """
        对 patches 进行 embedding。
        
        Args:
            patches: [B, N, p] patch 序列
            p: patch 长度
            
        Returns:
            [B, N, D] token 序列
        """
        w = self.get_weight(p)          # [D,p]
        b = self.base.bias              # [D]
        return F.linear(patches, w, b)  # [B,N,D]


class AttnPool(nn.Module):
    """
    注意力池化模块。
    
    使用可学习的 query 向量对 token 序列进行加权平均。
    
    Args:
        d_model: token 维度
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]
            
        Returns:
            [B, D] 池化后的表示
        """
        D = tokens.size(-1)
        scores = (tokens * self.q).sum(-1) / math.sqrt(D)   # [B,N]
        w = torch.softmax(scores, dim=1).unsqueeze(-1)      # [B,N,1]
        return (tokens * w).sum(dim=1)                      # [B,D]


class GatingNet(nn.Module):
    """
    门控网络。
    
    使用 MLP 对门控特征进行处理，输出 softmax 归一化的权重。
    
    Args:
        in_dim: 输入特征维度（默认 1 for ACF）
        hidden: 隐藏层维度
    """
    def __init__(self, in_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, feat: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Args:
            feat: [B, C, M, F] 门控特征
            tau: 温度参数
            
        Returns:
            [B, C, M] softmax 归一化的权重
        """
        logits = self.mlp(feat).squeeze(-1)      # [B,C,M]
        return torch.softmax(logits / tau, dim=-1)


class ChannelPeriodFlexEnhancer(nn.Module):
    """
    通道独立周期增强模块。
    
    将时间序列从 [B, L, C] 转换为 [B, C, D]，每个通道独立处理。
    使用多周期分支 + ACF 门控融合。
    
    Args:
        d_model: 输出 token 维度（对齐 iTransformer 的 d_model）
        p_ref: 母核 patch 长度
        n_fix: token 序列对齐长度
        m_periods: 每通道候选周期数
        period_table: [C, M] 周期表（可选）
        pool_type: 池化类型 "mean" 或 "attn"
        gate_hidden: 门控网络隐藏层维度
        tau: 门控 softmax 温度
        min_p: 最小周期
        max_p: 最大周期（可选）
    """
    def __init__(
        self,
        d_model: int,
        p_ref: int,
        n_fix: int,
        m_periods: int,
        period_table: Optional[torch.Tensor] = None,  # [C,M]
        pool_type: str = "mean",  # "mean" or "attn"
        gate_hidden: int = 32,
        tau: float = 1.0,
        min_p: int = 2,
        max_p: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.p_ref = int(p_ref)
        self.n_fix = int(n_fix)
        self.m = int(m_periods)
        self.tau = float(tau)
        self.min_p = int(min_p)
        self.max_p = max_p  # may depend on L; clamp in forward

        self.flex_embed = FlexPatchEmbedding(d_model=self.d_model, p_ref=self.p_ref)
        self.gating = GatingNet(in_dim=1, hidden=gate_hidden)
        self.pool_type = pool_type
        self.pool = AttnPool(self.d_model) if pool_type == "attn" else None

        if period_table is not None:
            assert period_table.dim() == 2 and period_table.size(1) == self.m, "period_table must be [C,M]"
            self.register_buffer("period_table", period_table.long())
        else:
            self.period_table = None

    def _clamp_period(self, p: int, L: int) -> int:
        """将周期限制在合法范围内。"""
        p = int(p)
        p = max(p, self.min_p)
        if self.max_p is not None:
            p = min(p, int(self.max_p))
        p = min(p, L)  # avoid p>L explosion; patchify has fallback but keep sane
        return p

    def clear_cache(self):
        """清理 FlexPatchEmbedding 的权重缓存。"""
        self.flex_embed.clear_cache()

    def forward(self, x: torch.Tensor, periods_override: Optional[torch.Tensor] = None, return_aux: bool = False):
        """
        前向传播。
        
        Args:
            x: [B, L, C] 输入时间序列
            periods_override: [C, M] 或 [B, C, M] 可选的周期覆盖
            return_aux: 是否返回辅助信息
            
        Returns:
            Z: [B, C, D] 增强后的 token
            aux: dict（当 return_aux=True）
                - periods_used: [C, M] 使用的周期
                - gating: [B, C, M] 门控权重
        """
        assert x.dim() == 3, "x must be [B,L,C]"
        B, L, C = x.shape

        # determine periods_used
        if periods_override is not None:
            periods_used = periods_override
        elif self.period_table is not None:
            periods_used = self.period_table
        else:
            raise ValueError("No periods provided. Set period_table or pass periods_override.")

        # normalize periods_used to [C,M] (MVP). If [B,C,M], handle per-sample later.
        if periods_used.dim() == 3:
            # MVP: take per-batch first item; or implement per-sample path if needed
            periods_used = periods_used[0]
        assert periods_used.shape == (C, self.m), f"periods_used must be [C,M], got {periods_used.shape}"

        Z_list = []
        gating_all = []

        for c in range(C):
            x_c = x[:, :, c]  # [B,L]
            z_pm = []
            feat_pm = []

            for mi in range(self.m):
                p = int(periods_used[c, mi].item())
                p = self._clamp_period(p, L)

                patches = patchify_1d(x_c, p)             # [B,N,p]
                tokens = self.flex_embed(patches, p)       # [B,N,D]
                tokens = resize_tokens(tokens, self.n_fix) # [B,n_fix,D]

                # pooling -> [B,D]
                if self.pool_type == "attn":
                    z = self.pool(tokens)
                else:
                    z = tokens.mean(dim=1)
                z_pm.append(z)

                # gating feature: acf@p -> [B,1]
                acf = acf_at_lag(x_c, p).unsqueeze(-1)     # [B,1]
                feat_pm.append(acf)

            # stack along M
            z_pm = torch.stack(z_pm, dim=1)            # [B,M,D]
            feat_pm = torch.stack(feat_pm, dim=1)      # [B,M,1]

            # gating per channel: [B,1,M,1] -> [B,1,M] -> squeeze -> [B,M]
            feat_in = feat_pm.unsqueeze(1)             # [B,1,M,1]
            g = self.gating(feat_in, tau=self.tau).squeeze(1)  # [B,M]
            gating_all.append(g)

            # fuse: [B,D]
            z_c = (z_pm * g.unsqueeze(-1)).sum(dim=1)   # [B,D]
            Z_list.append(z_c)

        Z = torch.stack(Z_list, dim=1)               # [B,C,D]
        if return_aux:
            gating_tensor = torch.stack(gating_all, dim=1)  # [B,C,M]
            aux = {
                "periods_used": periods_used,        # [C,M]
                "gating": gating_tensor,             # [B,C,M]
            }
            return Z, aux
        return Z
