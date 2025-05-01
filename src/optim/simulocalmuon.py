# file: full_momentum_loop_muon.py
# Author: your_name
# ----------------------------------------------------------------------
# Muon for plain DDP *without* any post-gradient communication.
#
#   • Each GPU owns the full weight and the full momentum buffer.
#   • Gradients are all-reduced once (global mean).
#   • For every column-stride shard (0 … world-1) we:
#         1) take momentum slice,
#         2) run 5-step Newton–Schulz,
#         3) apply update to that slice of W.
#   • All GPUs perform the *exact* same computation, so parameters
#     remain bit-identical across replicas — no gather/broadcast needed.
#
# Communication   : 1 × all_reduce (same as AdamW)
# Extra FLOPs     : × world_size  (each shard computed on every GPU)
# Extra memory    : full momentum buffer (like single-GPU Muon)
# ----------------------------------------------------------------------

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


# ---------------- Newton–Schulz helper (5 iterations) ---------------- #
def zeropower_via_newtonschulz5(G: torch.Tensor,
                                steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Return (G Gᵀ)^−½ G approximated with 5 Newton–Schulz iterations."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G if G.size(0) <= G.size(1) else G.t()   # make cols ≥ rows
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    return X if G.size(0) <= G.size(1) else X.t()
# --------------------------------------------------------------------- #


class simulocalmuon(Optimizer):
    """Full-momentum Muon with shard-loop update under DataParallel."""

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 momentum: float = 0.95,
                 ns_steps: int = 5,
                 betas: tuple = (0.95, 0.95),        # AdamW for 1-D
                 eps: float = 1e-8,
                 weight_decay: float = 0.1,
                 dp_group: Optional[dist.ProcessGroup] = None):

        defaults = dict(lr=lr, momentum=momentum, betas=betas,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.ns_steps = ns_steps
        self.dp_group = dp_group
        self.world = dist.get_world_size(dp_group)
        self.rank = dist.get_rank(dp_group)

        # allocate empty state dicts; we lazily fill on first step
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {}                 # lazy init for all types

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # ---- 0. all-reduce to global mean gradient -------------
                dist.all_reduce(p.grad, group=self.dp_group)
                p.grad.div_(self.world)

                state = self.state[p]

                # ======== MATRIX PARAMETERS → Muon update ===============
                if p.ndim >= 2:
                    # lazy-init full momentum buffer
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.data)

                    M = state["momentum"]
                    M.mul_(mu).add_(p.grad)        # update full momentum

                    rows, cols = p.shape
                    scale = 0.2 * math.sqrt(max(rows, cols))  # RMS match

                    # iterate over column-stride shards
                    for s in range(self.world):
                        col_slice = slice(s, cols, self.world)
                        mom_slice = M[:, col_slice]
                        O = zeropower_via_newtonschulz5(mom_slice,
                                                        self.ns_steps)
                        delta = -lr * scale * O

                        W_slice = p.data[:, col_slice]
                        W_slice.mul_(1 - lr * wd).add_(delta)

                # ======== 1-D PARAMETERS → AdamW =======================
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad,
                                                    value=1 - beta2)

                    bias_c1 = 1 - beta1 ** state["step"]
                    bias_c2 = 1 - beta2 ** state["step"]
                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_c2)).add_(eps)

                    step_size = lr / bias_c1
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
# ----------------------------------------------------------------------
# Quick usage:
#
# >>> dist.init_process_group("nccl")
# >>> model = MyModel().cuda()
# >>> model = torch.nn.parallel.DistributedDataParallel(model)
# >>> optim = FullMomentumLoopMuon(model.parameters(), lr=1e-3)
# >>> ...
# ----------------------------------------------------------------------
