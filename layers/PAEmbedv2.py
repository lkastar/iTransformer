import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 核心工具函数 (针对 Seasonal 优化)
# ==========================================

def resample_patchemb(old_weight: torch.Tensor, new_patch_len: int):
    """
    Flex-Resize: 将参考权重缩放到目标周期长度 P
    """
    assert old_weight.dim() == 2
    if old_weight.size(1) == new_patch_len:
        return old_weight
    old = old_weight.T
    old_len = old.size(0)
    factor = new_patch_len / old_len
    basis_vectors = torch.eye(old_len, dtype=torch.float32, device=old.device)
    resize_mat = F.interpolate(basis_vectors.unsqueeze(0).unsqueeze(0), 
                               size=(new_patch_len, old_len), 
                               mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)
    resampled_kernels = resize_mat_pinv @ old * math.sqrt(factor)
    return resampled_kernels.T

def ACF_for_Seasonal(x, k=2, min_period=4, acf_threshold=0.15):
    """
    针对 Seasonal 成分提取 Top-K 周期
    """
    B, T, C = x.shape
    x_centered = x - x.mean(dim=1, keepdim=True)
    n_fft = 1 << (2 * T - 1).bit_length()
    xf = torch.fft.rfft(x_centered, n=n_fft, dim=1)
    acf = torch.fft.irfft(xf * torch.conj(xf), n=n_fft, dim=1)[:, :T, :]
    avg_acf = acf.mean(dim=0) # [T, C]
    
    # 归一化 ACF 以应用阈值过滤器
    norm_acf = avg_acf / (avg_acf[0, :] + 1e-9)
    norm_acf[:min_period, :] = -float('inf')
    
    prev_lag = torch.roll(norm_acf, 1, dims=0)
    next_lag = torch.roll(norm_acf, -1, dims=0)
    # NOTE: 识别峰值且必须超过阈值，增强对 Seasonal 噪声的鲁棒性
    is_peak = (norm_acf > prev_lag) & (norm_acf > next_lag) & (norm_acf > acf_threshold)
    
    masked_acf = norm_acf.clone()
    masked_acf[~is_peak] = -float('inf')
    
    top_vals, top_inds = torch.topk(masked_acf, k, dim=0)
    # 若无显著周期，回退到全长 T
    final_periods = torch.where(top_vals > -float('inf'), top_inds, torch.full_like(top_inds, T))
    return final_periods.t() # [C, k]

# ==========================================
# 2. 核心模块: Seasonal Aggregator
# ==========================================

class SeasonalAggregator(nn.Module):
    """
    利用最后两个 Patch 的均值作为 Query，增强季节性相位的稳定性
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches):
        # patches: [B * C_subset, Num_Patches, d_model]
        if patches.size(1) >= 2:
            # NOTE: 同步平均的进阶版：相位敏感 Query
            query = (patches[:, -1:, :] + patches[:, -2:-1, :]) / 2.0
        else:
            query = patches[:, -1:, :]
            
        attn_out, _ = self.attn(query, patches, patches)
        return self.norm(query + self.dropout(attn_out)).squeeze(1)

# ==========================================
# 3. PAEmbedding_Seasonal (无需 c_in)
# ==========================================

class PAEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, anchor_periods=[12, 24, 48]):
        """
        Seasonal 专用 PAEmbedding，完全通道独立，不依赖输入变量数
        """
        super(PAEmbedding, self).__init__()
        self.d_model = d_model
        self.anchor_periods = sorted(anchor_periods)
        
        # 多锚点权重库
        self.anchor_weights = nn.ParameterDict()
        for p in self.anchor_periods:
            w = nn.Parameter(torch.empty(d_model, p))
            nn.init.xavier_uniform_(w)
            self.anchor_weights[str(p)] = w
            
        self.shared_bias = nn.Parameter(torch.zeros(d_model))
        
        # 聚合器
        self.agg_p1 = SeasonalAggregator(d_model, dropout=dropout)
        self.agg_p2 = SeasonalAggregator(d_model, dropout=dropout)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.final_dropout = nn.Dropout(p=dropout)
        self._weight_cache = {} 

    def _get_flex_weight(self, target_len):
        if target_len in self._weight_cache:
            return self._weight_cache[target_len]
        nearest_p = min(self.anchor_periods, key=lambda x: abs(x - target_len))
        new_w = resample_patchemb(self.anchor_weights[str(nearest_p)], target_len)
        self._weight_cache[target_len] = new_w
        return new_w

    def forward(self, x_s):
        """
        x_s: [B, T, C] -> EMA 分解后的 Seasonal 成分
        """
        B, T, C = x_s.shape
        self._weight_cache = {}

        # 1. 周期识别
        with torch.no_grad():
            periods = ACF_for_Seasonal(x_s, k=2) 
            periods = torch.clamp(periods, min=4, max=T)
        
        # 2. 转置以符合通道独立逻辑
        # NOTE: x_inverted 为 [B, C, T]，所有后续操作均在 T 维度
        x_inverted = x_s.permute(0, 2, 1) 
        
        # --- Stream 3: Global Seasonal Reference (原 Stream 3 回归) ---
        # NOTE: 针对 Seasonal 的全长度进行 Flex-Resize 映射
        w_trend = self._get_flex_weight(T) 
        stream_global_out = F.linear(x_inverted, w_trend, self.shared_bias)

        # --- Stream 1 & 2: Period Streams ---
        stream_p1_out = torch.zeros(B, C, self.d_model, device=x_s.device)
        stream_p2_out = torch.zeros(B, C, self.d_model, device=x_s.device)
        
        unique_periods = torch.unique(periods)
        for p in unique_periods:
            p_val = p.item()
            if p_val >= T: continue # 回退逻辑：不进行 Patch 切分
            
            mask_p1 = (periods[:, 0] == p_val)
            mask_p2 = (periods[:, 1] == p_val)
            if not (mask_p1.any() or mask_p2.any()): continue
                
            w_p = self._get_flex_weight(p_val)
            target_channels = mask_p1 | mask_p2
            x_sub = x_inverted[:, target_channels, :] 
            
            # NOTE: 在 T 维度上切分 Seasonal Patch
            patches = x_sub.unfold(dimension=2, size=p_val, step=p_val) 
            num_p = patches.size(2)
            
            # NOTE: Tokenize
            tokens = F.linear(patches, w_p, self.shared_bias)
            
            for i, (m, agg) in enumerate([(mask_p1, self.agg_p1), (mask_p2, self.agg_p2)]):
                if m.any():
                    rel_idx = torch.isin(torch.where(target_channels)[0], torch.where(m)[0])
                    # NOTE: 合并 B*C 进行并行聚合，确保通道独立
                    sub_tokens = tokens[:, rel_idx, :, :].reshape(-1, num_p, self.d_model)
                    agg_res = agg(sub_tokens).reshape(B, -1, self.d_model)
                    
                    if i == 0: stream_p1_out[:, m, :] = agg_res
                    else: stream_p2_out[:, m, :] = agg_res

        # --- Fusion ---
        combined = torch.cat([stream_p1_out, stream_p2_out, stream_global_out], dim=-1)
        output = self.fusion_layer(combined) # [B, C, d_model]
        return self.final_dropout(output)