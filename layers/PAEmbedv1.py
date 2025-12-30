import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 核心工具函数 (保持不变)
# ==========================================

def resample_patchemb(old_weight: torch.Tensor, new_patch_len: int):
    assert old_weight.dim() == 2
    if old_weight.size(1) == new_patch_len:
        return old_weight
    old = old_weight.T
    old_len = old.size(0)
    factor = new_patch_len / old_len
    basis_vectors = torch.eye(old_len, dtype=torch.float32, device=old.device)
    resize_mat = F.interpolate(basis_vectors.unsqueeze(0).unsqueeze(0), size=(new_patch_len, old_len), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)
    resampled_kernels = resize_mat_pinv @ old * math.sqrt(factor)
    return resampled_kernels.T

def ACF_for_Period_Per_Channel(x, k=2, min_period=4):
    """
    x: [B, T, C]
    返回: [C, k] tensor，为每个通道独立计算周期
    """
    B, T, C = x.shape
    x_centered = x - x.mean(dim=1, keepdim=True)
    n_fft = 1 << (2 * T - 1).bit_length()
    xf = torch.fft.rfft(x_centered, n=n_fft, dim=1)
    acf = torch.fft.irfft(xf * torch.conj(xf), n=n_fft, dim=1)[:, :T, :]
    avg_acf = acf.mean(dim=0) # [T, C]
    avg_acf[:min_period, :] = -float('inf')
    prev_lag = torch.roll(avg_acf, 1, dims=0)
    next_lag = torch.roll(avg_acf, -1, dims=0)
    is_peak = (avg_acf > prev_lag) & (avg_acf > next_lag) & (avg_acf > 0)
    masked_acf = avg_acf.clone()
    masked_acf[~is_peak] = -float('inf')
    top_vals, top_inds = torch.topk(masked_acf, k, dim=0)
    return top_inds.t() # [C, k]

# ==========================================
# 2. 核心模块: Attentive Aggregator
# ==========================================

class AttentiveAggregator(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches):
        """
        patches: [Total_Tokens, Num_Patches, d_model] 
        # NOTE: 此处的 Total_Tokens = B * C_subset，即所有通道独立处理
        """
        query = patches[:, -1:, :] 
        attn_out, _ = self.attn(query, patches, patches)
        return self.norm(query + self.dropout(attn_out)).squeeze(1)

# ==========================================
# 3. 适配 iTransformer 的 PAEmbedding
# ==========================================

class PAEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, anchor_periods=[12, 24, 48, 96]):
        """
        c_in: 在 iTransformer 中通常对应 Lookback Window 长度 (seq_len)
        d_model: 嵌入维度
        """
        super(PAEmbedding, self).__init__()
        self.d_model = d_model
        self.anchor_periods = sorted(anchor_periods)
        
        self.anchor_weights = nn.ParameterDict()
        for p in self.anchor_periods:
            w = nn.Parameter(torch.empty(d_model, p))
            nn.init.xavier_uniform_(w)
            self.anchor_weights[str(p)] = w
            
        self.shared_bias = nn.Parameter(torch.zeros(d_model))
        self.agg_p1 = AttentiveAggregator(d_model, dropout=dropout)
        self.agg_p2 = AttentiveAggregator(d_model, dropout=dropout)
        
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

    def forward(self, x):
        """
        x: [B, T, C] -> 期望输入形状
        """
        B, T, C = x.shape
        self._weight_cache = {}

        # 1. 周期识别: 基于时间轴 T 为每个通道独立计算 [C, 2]
        with torch.no_grad():
            periods = ACF_for_Period_Per_Channel(x, k=2) 
            periods = torch.clamp(periods, min=4, max=T//2)
        
        # 2. 转置以符合 iTransformer 通道独立逻辑
        # NOTE: 转换 x 为 [B, C, T]，后续所有特征提取均在 T 维度操作，保持 C 独立
        x_inverted = x.permute(0, 2, 1) # [B, C, T]
        
        # --- Stream 3: 全局趋势流 (Global Trend) ---
        w_trend = self._get_flex_weight(T) 
        # NOTE: F.linear 作用于 x_inverted 的最后一个维度 T，输出 [B, C, d_model]
        stream_trend_out = F.linear(x_inverted, w_trend, self.shared_bias)

        # --- Stream 1 & 2: 周期模式流 (Period Patterns) ---
        stream_p1_out = torch.zeros(B, C, self.d_model, device=x.device)
        stream_p2_out = torch.zeros(B, C, self.d_model, device=x.device)
        
        unique_periods = torch.unique(periods)
        for p in unique_periods:
            p_val = p.item()
            mask_p1 = (periods[:, 0] == p_val)
            mask_p2 = (periods[:, 1] == p_val)
            if not (mask_p1.any() or mask_p2.any()): continue
                
            w_p = self._get_flex_weight(p_val)
            target_channels = mask_p1 | mask_p2
            
            # NOTE: 仅提取符合当前周期的通道，数据形状为 [B, C_subset, T]
            x_sub = x_inverted[:, target_channels, :] 
            
            # NOTE: 在 T 维度上进行非重叠切片 (Patching)，不改变 B 和 C 维度
            # patches shape: [B, C_subset, Num_Patches, P]
            patches = x_sub.unfold(dimension=2, size=p_val, step=p_val) 
            num_p = patches.size(2)
            
            # NOTE: Tokenize 操作作用于 Patch 长度 P，映射至 d_model
            # tokens shape: [B, C_subset, Num_Patches, d_model]
            tokens = F.linear(patches, w_p, self.shared_bias)
            
            for i, (m, agg) in enumerate([(mask_p1, self.agg_p1), (mask_p2, self.agg_p2)]):
                if m.any():
                    # NOTE: 找到当前周期槽位（P1或P2）在 x_sub 中的相对通道索引
                    rel_idx = torch.isin(torch.where(target_channels)[0], torch.where(m)[0])
                    # NOTE: 合并 B 和 C 维度，在通道独立的条件下聚合 Num_Patches 维度的时序信息
                    # input to agg: [B * C_p_subset, Num_Patches, d_model]
                    sub_tokens = tokens[:, rel_idx, :, :].reshape(-1, num_p, self.d_model)
                    agg_res = agg(sub_tokens).reshape(B, -1, self.d_model) # [B, C_p_subset, d_model]
                    
                    if i == 0: stream_p1_out[:, m, :] = agg_res
                    else: stream_p2_out[:, m, :] = agg_res

        # --- Fusion & Output ---
        # NOTE: 在 d_model 维度拼接三个流，保持 B 和 C 维度不变
        combined = torch.cat([stream_p1_out, stream_p2_out, stream_trend_out], dim=-1)
        # NOTE: 融合层在每个通道 token 上独立进行 MLP 映射
        output = self.fusion_layer(combined) # [B, C, d_model]
        return self.final_dropout(output)
    
import torch

def test_pa_embedding():
    # ---------------------------------------------------------
    # 1. 模拟超参数 (对齐 iTransformer 配置文件)
    # ---------------------------------------------------------
    B = 4          # Batch Size
    T = 96         # Lookback Window (seq_len)
    C = 7          # Number of Variates (Channels)
    d_model = 512  # Embedding Dimension
    
    print(f"--- 测试开始 ---")
    print(f"输入参数: Batch={B}, Time={T}, Channels={C}, d_model={d_model}")

    # 实例化 PAEmbedding
    # 注意: c_in 在 iTransformer 调用中实际上指的是输入的 seq_len
    embed_layer = PAEmbedding(d_model=d_model, dropout=0.1)
    
    # ---------------------------------------------------------
    # 2. 模拟输入数据 [B, T, C]
    # ---------------------------------------------------------
    # 我们为不同通道构造不同的周期特性，验证 ACF 是否能识别
    x = torch.zeros(B, T, C)
    t = torch.arange(T).float()
    
    for c in range(C):
        if c % 2 == 0:
            # 偶数通道：强周期性 (周期约 24)
            x[:, :, c] = torch.sin(2 * math.pi * t / 24) + torch.randn(B, T) * 0.1
        else:
            # 奇数通道：线性趋势 + 噪声 (ACF 应回退到全局趋势)
            x[:, :, c] = 0.05 * t + torch.randn(B, T) * 0.1
            
    print(f"输入形状: {x.shape} (Expected: [B, T, C])")

    # ---------------------------------------------------------
    # 3. 执行前向传播
    # ---------------------------------------------------------
    try:
        output = embed_layer(x)
        print(f"输出形状: {output.shape} (Expected: [{B}, {C}, {d_model}])")
        
        # 验证维度
        assert output.shape == (B, C, d_model), "维度检查失败！"
        print("✅ 维度校验通过。")
        
    except Exception as e:
        print(f"❌ 前向传播出错: {e}")
        return

    # ---------------------------------------------------------
    # 4. 验证通道独立性 (Channel Independence)
    # ---------------------------------------------------------
    # 逻辑：修改 x 中 Channel 0 的数据，Channel 1 的输出 Token 不应改变
    x_modified = x.clone()
    x_modified[:, :, 0] += 10.0 # 剧烈改变通道 0
    
    output_orig = embed_layer(x)
    output_mod = embed_layer(x_modified)
    
    # 检查通道 1 的 Token 是否一致 (由于 ACF 识别可能有微弱随机性，我们检查数值差异)
    diff = torch.abs(output_orig[:, 1, :] - output_mod[:, 1, :]).max()
    
    if diff < 1e-5:
        print(f"✅ 通道独立性校验通过 (Diff={diff:.2e})。修改通道0不影响通道1。")
    else:
        print(f"⚠️ 通道独立性存在风险 (Diff={diff:.2e})。请检查缓存或全局操作。")

    # ---------------------------------------------------------
    # 5. 权重缓存校验
    # ---------------------------------------------------------
    print(f"当前权重缓存条数: {len(embed_layer._weight_cache)}")
    print(f"--- 测试结束 ---")

if __name__ == "__main__":
    # 为了运行测试，确保环境中已定义上文提到的 PAEmbedding, ACF_for_Period_Per_Channel 等类
    import math 
    test_pa_embedding()