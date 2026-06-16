"""clearn.lora — manual LoRA injection.

LoRA is implemented by hand (not peft) because in the full system the low-rank
factors are *generated* by a hypernetwork, not learned as free parameters. A
``LoRALinear`` therefore supports two modes:

  * ``set_params(A, B)`` with leaf nn.Parameters  -> oracle / directly-optimized
  * ``set_delta(A, B)``  with externally-supplied tensors -> hypernetwork output

Delta:  y = W x  +  scaling * (x A^T) B^T ,  with A:[r,in], B:[out,r].

A third, *spectral* mode anchors the adaptation to the base weight's own SVD
(W = U S V^T) instead of free factors. We freeze the top-r1 + bottom-r2 singular
triplets and learn only a small r x r core R (r = r1+r2):

    Delta_W = U_r diag(S_r) R V_r^T ,   R:[r,r] learnable (zero-init).

This never adds directions the base model did not already have (it only mixes
within the chosen singular subspace), and R is tiny -> cheap to learn/generate.
  * ``set_spectral(device, dtype)``     -> learnable R (oracle)
  * ``set_spectral_delta(R)``           -> externally-supplied R (hypernetwork)
"""
from __future__ import annotations
import torch, torch.nn as nn

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float | None = None, mode: str = "lora"):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.in_f, self.out_f = base.in_features, base.out_features
        self.alpha = alpha
        self.scaling = (alpha if alpha is not None else r) / r
        self.mode = mode  # "lora" | "spectral"
        self.A = None  # [r, in]
        self.B = None  # [out, r]
        self.R = None  # [r, r]   (spectral core)

    def set_params(self, device, dtype):
        """Allocate learnable A,B (oracle mode). A ~ small, B = 0 -> zero init delta."""
        self.A = nn.Parameter(torch.randn(self.r, self.in_f, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_f, self.r, device=device, dtype=dtype))
        return [self.A, self.B]

    def set_delta(self, A, B):
        """Inject externally-generated factors (hypernetwork mode)."""
        self.A, self.B = A, B

    def setup_spectral(self, r1: int, r2: int = 0):
        """Freeze the top-r1 + bottom-r2 singular basis of the base weight.

        Registers buffers P = U_r diag(S_r)  [out, r]  and  Vt = V_r^T  [r, in],
        so Delta_W = P R Vt and the injected term is  ((x @ Vt^T) @ R^T) @ P^T.
        """
        W = self.base.weight.data.float()                       # [out, in]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)      # U[out,k] S[k] Vh[k,in]
        k = S.shape[0]
        if r1 + r2 > k:
            raise ValueError(f"r1+r2={r1 + r2} exceeds rank k={k}")
        top = torch.arange(r1)
        bot = torch.arange(k - r2, k) if r2 else torch.arange(0)
        idx = torch.cat([top, bot])
        P = (U[:, idx] * S[idx]).contiguous()                    # [out, r]
        Vt = Vh[idx].contiguous()                                # [r, in]
        self.register_buffer("P", P, persistent=False)
        self.register_buffer("Vt", Vt, persistent=False)
        self.r = r1 + r2
        self.scaling = (self.alpha if self.alpha is not None else self.r) / self.r
        self.mode = "spectral"
        return self

    def set_spectral(self, device, dtype):
        """Oracle: learnable r x r core R (zero init -> zero initial delta)."""
        self.R = nn.Parameter(torch.zeros(self.r, self.r, device=device, dtype=dtype))
        return [self.R]

    def set_spectral_delta(self, R):
        """Inject an externally-generated core R (hypernetwork mode). R:[...,r,r]."""
        self.R = R

    def clear(self):
        self.A = self.B = self.R = None

    def forward(self, x):
        y = self.base(x)
        if self.mode == "spectral":
            if self.R is None:
                return y
            R = self.R.to(x.dtype)
            P, Vt = self.P.to(x.dtype), self.Vt.to(x.dtype)
            xV = torch.matmul(x, Vt.transpose(-1, -2))            # [..., r]
            xVR = torch.matmul(xV, R.transpose(-1, -2))           # [..., r]
            delta = torch.matmul(xVR, P.transpose(-1, -2))        # [..., out]
            return y + self.scaling * delta
        if self.A is None:
            return y
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        delta = torch.matmul(torch.matmul(x, A.transpose(-1, -2)), B.transpose(-1, -2))
        return y + self.scaling * delta


def inject(model, targets=("q_proj", "v_proj"), r=8, alpha=None, mode="lora"):
    """Replace target nn.Linear submodules with LoRALinear. Returns the list."""
    wrapped = []
    for layer in model.model.layers:
        for name in targets:
            parent = layer.self_attn if name in ATTN else layer.mlp
            mod = getattr(parent, name)
            if isinstance(mod, LoRALinear):
                mod = mod.base
            lin = LoRALinear(mod, r=r, alpha=alpha, mode=mode)
            setattr(parent, name, lin)
            wrapped.append(lin)
    return wrapped


def remove(model):
    """Restore original nn.Linear (undo inject)."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, mod in list(parent.named_children()):
                if isinstance(mod, LoRALinear):
                    setattr(parent, name, mod.base)


def alloc_oracle(wrapped, device, dtype):
    params = []
    for w in wrapped:
        params += w.set_params(device, dtype)
    return params


def alloc_spectral(wrapped, r1, r2, device, dtype):
    """Spectral oracle: freeze each module's SVD basis, learn its r x r core R."""
    params = []
    for w in wrapped:
        w.setup_spectral(r1, r2)
        params += w.set_spectral(device, dtype)
    return params
