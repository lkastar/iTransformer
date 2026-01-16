## Plan: `TokenSemanticAugmenter` Module (inputs: x, mu, sigma; outputs: [B,C,d_model])

### 0) Objectives & Constraints

**Objective**

* Implement a PyTorch module that takes:

  * `x`: time series tensor, shape `[B,L,C]`
  * `mu`, `sigma`: ReVIN cache tensors (prefer `[B,C]`, optionally `[B,L,C]`)
* Produces:

  * `tokens`: enhanced variable tokens, shape `[B,C,d_model]`

**Constraints**

* Avoid multiscale / patch pyramids.
* Use Option B: tiny intra-token attention summarizer on time dimension.
* Add lightweight dynamics-aware semantics (AR proxy, kinematics, acf, residual energy, scale semantics).
* All operations must be vectorized and differentiable.
* Provide clear shape annotations and robust input validation.

---

## 1) File & Class Skeleton

### 1.1 File

* Create `token_semantic_augmenter.py`

### 1.2 Class signature

Implement:

```python
class TokenSemanticAugmenter(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_intra: int = 64,
        ar_order: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        eps: float = 1e-5,
        use_skip: bool = True,
        use_dyn: bool = True,
        use_ar: bool = True,
        use_acf: bool = True,
        use_scale: bool = True,
        ar_learnable: bool = True,
    ):
        ...
    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

---

## 2) Input Validation & Preprocessing

### 2.1 Validate shapes

* assert `x.ndim == 3` and `x.shape == [B,L,C]`
* mu/sigma must be either:

  * `[B,C]` or
  * `[B,L,C]` (compat mode)

Implement:

* if `mu.ndim==3`: reduce `mu = mu.mean(dim=1)` → `[B,C]`
* if `sigma.ndim==3`: reduce `sigma = sigma.mean(dim=1)` OR (better) compute along L from original cache if available

  * In compat mode: `sigma = sigma.mean(dim=1)` acceptable (cache should be constant over L anyway)

### 2.2 Require minimum L

This module uses:

* AR residual (needs `L >= ar_order`)
* acf2 (needs `L >= 3`)

Add checks:

* `L >= max(ar_order + 1, 3)` (safe for residual alignment + acf2)
* Provide informative error messages.

### 2.3 Normalize for feature computation

Assume `x` is already normalized by ReVIN outside (common), but we cannot rely on it.
Therefore:

* Compute a normalized version `xn` using the provided cache:

  * `xn = (x - mu[:,None,:]) / (sigma[:,None,:] + eps)` → `[B,L,C]`

This ensures the module is self-consistent even if caller passes raw `x`.

---

## 3) Layout Conversion

Convert for per-variable processing:

* `U = xn.transpose(1,2)` → `[B,C,L]`

Also keep raw-scale series (optional for some features):

* `U_raw = x.transpose(1,2)` → `[B,C,L]` (only if you want raw-based kinematics; default compute on normalized U)

---

## 4) Step A — Tiny Intra-token Attention Summarizer (Option B)

### 4.1 Project scalar to `d_intra`

Define:

* `self.intra_proj = nn.Linear(1, d_intra, bias=True)`
* `self.query = nn.Parameter(torch.randn(d_intra))`
* `self.base_ln = nn.LayerNorm(d_intra)`

Compute:

* `H = intra_proj(U.unsqueeze(-1))` → `[B,C,L,d_intra]`

### 4.2 Attention pooling with learnable query

* `q = self.query` → `[d_intra]`
* `scores = (H * q).sum(dim=-1) / sqrt(d_intra)` → `[B,C,L]`
* `alpha = softmax(scores, dim=-1)` → `[B,C,L]`
* `E_base = (alpha.unsqueeze(-1) * H).sum(dim=2)` → `[B,C,d_intra]`
* `E_base = base_ln(E_base)` for stability.

---

## 5) Step B — Dynamics-aware Feature Block (no multiscale)

This block outputs `dyn ∈ [B,C,k]` via concatenating several cheap features. It must be switchable via flags.

### 5.1 Kinematics features (always on if use_dyn)

Compute on `U` (normalized) by default:

* `dx = U[:,:,1:] - U[:,:,:-1]` → `[B,C,L-1]`
* `ddx = dx[:,:,1:] - dx[:,:,:-1]` → `[B,C,L-2]`

Features:

* `dx_mean = dx.mean(dim=-1)` → `[B,C]`
* `dx_var  = dx.var(dim=-1, unbiased=False)` → `[B,C]`
* `ddx_mean = ddx.mean(dim=-1)` → `[B,C]`

### 5.2 ACF proxies (optional, use_acf)

Compute centered series:

* `Uc = U - U.mean(dim=-1, keepdim=True)` → `[B,C,L]`
* `den = Uc.pow(2).mean(dim=-1) + eps` → `[B,C]`

Lag1:

* `num1 = (Uc[:,:,1:] * Uc[:,:,:-1]).mean(dim=-1)` → `[B,C]`
* `acf1 = num1 / den` → `[B,C]`

Lag2:

* `num2 = (Uc[:,:,2:] * Uc[:,:,:-2]).mean(dim=-1)` → `[B,C]`
* `acf2 = num2 / den` → `[B,C]`

### 5.3 AR proxy residual energy (optional, use_ar)

Implement a grouped conv1d AR predictor per variable:

Define:

* `self.ar_conv = nn.Conv1d(
      in_channels=C, out_channels=C,
      kernel_size=ar_order, groups=C, bias=False
  )`
* If `ar_learnable=False`, set `requires_grad_(False)` for weights.

Compute:

* input: `U` is `[B,C,L]`
* `pred = ar_conv(U)` → `[B,C,L-ar_order+1]`
* target aligned: `target = U[:,:,ar_order-1:]` → `[B,C,L-ar_order+1]`
* residual: `res = target - pred` → same shape

Features:

* `res_var = res.var(dim=-1, unbiased=False)` → `[B,C]`
* `res_mean = res.mean(dim=-1)` optional (keep if you want)
* stability proxy from weights:

  * `w = ar_conv.weight` shape `[C,1,ar_order]`
  * `stab = w.abs().sum(dim=-1).squeeze(1)` → `[C]`
  * broadcast: `stab_b = stab.unsqueeze(0).expand(B, -1)` → `[B,C]`

### 5.4 Scale semantics from cache (optional, use_scale)

Even if `xn` is normalized, scale semantics are useful:

* `mu_feat = mu` → `[B,C]`
* `log_sigma = log(sigma + eps)` → `[B,C]`

### 5.5 Assemble dyn feature vector

Create a python list `feat_list` of tensors shaped `[B,C,1]`:

* always include (if use_dyn):

  * `dx_mean.unsqueeze(-1)`
  * `dx_var.unsqueeze(-1)`
  * `ddx_mean.unsqueeze(-1)`
* if use_acf:

  * `acf1.unsqueeze(-1)`
  * `acf2.unsqueeze(-1)`
* if use_ar:

  * `res_var.unsqueeze(-1)`
  * `stab_b.unsqueeze(-1)`
* if use_scale:

  * `mu_feat.unsqueeze(-1)`
  * `log_sigma.unsqueeze(-1)`

Concatenate:

* `dyn = torch.cat(feat_list, dim=-1)` → `[B,C,k]`

Normalize dyn:

* `self.dyn_ln = nn.LayerNorm(k)`
* `dyn = dyn_ln(dyn)`

Important: `k` depends on flags; you must compute `k` in `__init__` given flags, and create `LayerNorm(k)` accordingly.

---

## 6) Step C — Fusion Head to `d_model`

### 6.1 Concatenate base + dyn (or only base)

If `use_dyn`:

* `Z = cat([E_base, dyn], dim=-1)` → `[B,C,d_intra+k]`
  Else:
* `Z = E_base` → `[B,C,d_intra]`

### 6.2 Fusion MLP

Define `in_dim` accordingly (`d_intra` or `d_intra+k`).

Create:

* `self.fuse_ln = nn.LayerNorm(in_dim)`
* `hidden = int(mlp_ratio * d_model)`
* `self.fuse_mlp = nn.Sequential(
    nn.Linear(in_dim, hidden),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden, d_model),
    nn.Dropout(dropout),
  )`

Compute:

* `T = fuse_mlp(fuse_ln(Z))` → `[B,C,d_model]`

### 6.3 Optional skip connection

If `use_skip=True`:

* define `self.skip_proj = nn.Linear(d_intra, d_model, bias=False)`
* `T = T + skip_proj(E_base)`

Return `T`.

---

## 7) Initialization Details

* Initialize `query` with `normal_(0, 0.02)`
* Initialize `intra_proj` with default or Xavier.
* Initialize `ar_conv.weight` with small normal (0, 0.01) to stabilize early training.

---

## 8) Unit Tests (must implement)

Create `test_token_semantic_augmenter.py`:

### 8.1 Shape test

* `x = randn(2,96,7)`, `mu/sigma = randn(2,7), abs(randn)+1e-1`
* `T = module(x,mu,sigma)`
* assert `T.shape == (2,7,d_model)`

### 8.2 Compat mode test (`mu/sigma` as `[B,L,C]`)

* `mu3 = mu[:,None,:].expand(B,L,C)`
* `sigma3 = sigma[:,None,:].expand(B,L,C)`
* `T2 = module(x,mu3,sigma3)`
* assert shape ok and `torch.isfinite(T2).all()`

### 8.3 Backprop test

* `T.sum().backward()`
* ensure grads exist on `query` and `intra_proj.weight`
* if `use_ar & ar_learnable`: check `ar_conv.weight.grad` not None.

### 8.4 Length constraint test

* Try `L < ar_order+1`, expect `ValueError` with clear message.

---

## 9) Documentation Requirements

Add docstring describing:

* expected input shapes
* interpretation of `mu,sigma`
* what semantics are encoded in `dyn`
* notes about normalized computation (`xn`)

Add inline comments for each tensor shape at key steps.

---

## 10) Deliverable Summary

The implementing AI must output:

* `token_semantic_augmenter.py` containing the complete module
* `test_token_semantic_augmenter.py` with the tests above
* No integration code beyond module + tests
