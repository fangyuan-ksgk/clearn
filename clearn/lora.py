"""clearn.lora — manual LoRA injection.

LoRA is implemented by hand (not peft) because in the full system the low-rank
factors are *generated* by a hypernetwork, not learned as free parameters. A
``LoRALinear`` therefore supports two modes:

  * ``set_params(A, B)`` with leaf nn.Parameters  -> oracle / directly-optimized
  * ``set_delta(A, B)``  with externally-supplied tensors -> hypernetwork output

Delta:  y = W x  +  scaling * (x A^T) B^T ,  with A:[r,in], B:[out,r].
"""
from __future__ import annotations
import torch, torch.nn as nn

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.in_f, self.out_f = base.in_features, base.out_features
        self.scaling = (alpha if alpha is not None else r) / r
        self.A = None  # [r, in]
        self.B = None  # [out, r]

    def set_params(self, device, dtype):
        """Allocate learnable A,B (oracle mode). A ~ small, B = 0 -> zero init delta."""
        self.A = nn.Parameter(torch.randn(self.r, self.in_f, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_f, self.r, device=device, dtype=dtype))
        return [self.A, self.B]

    def set_delta(self, A, B):
        """Inject externally-generated factors (hypernetwork mode)."""
        self.A, self.B = A, B

    def clear(self):
        self.A = self.B = None

    def forward(self, x):
        y = self.base(x)
        if self.A is None:
            return y
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        delta = torch.matmul(torch.matmul(x, A.transpose(-1, -2)), B.transpose(-1, -2))
        return y + self.scaling * delta


def inject(model, targets=("q_proj", "v_proj"), r=8, alpha=None):
    """Replace target nn.Linear submodules with LoRALinear. Returns the list."""
    wrapped = []
    for layer in model.model.layers:
        for name in targets:
            parent = layer.self_attn if name in ATTN else layer.mlp
            mod = getattr(parent, name)
            if isinstance(mod, LoRALinear):
                mod = mod.base
            lin = LoRALinear(mod, r=r, alpha=alpha)
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
