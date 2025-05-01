# file: local_shard_muon_dp.py
# Author: your_name
# ----------------------------------------------------------------------
# A Data-Parallel implementation of Muon that mimics “Distributed Muon”
# (Algorithm 1 in “Muon is Scalable for LLM Training”) but without ZeRO-1.
#
#   1. gradients are all-reduced (global mean) because each GPU owns the
#      full parameter replica under DDP;
#   2. each rank takes the column-stride slice that corresponds to it
#      (rank r gets columns r, r+world, …);
#   3. Newton–Schulz orthogonalisation is applied on that slice together
#      with its own momentum shard;
#   4. only that slice is updated, then all slices are gathered on rank 0
#      and broadcast back so all replicas stay identical.
#
# Communication volume is higher than true ZeRO-1 (we transmit the whole
# gradient once), but we still cut momentum memory by 1/world_size.
# ----------------------------------------------------------------------

import math
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


# ---------------- Newton–Schulz helper (5 iterations) ---------------- #
def zeropower_via_newtonschulz5(G: torch.Tensor,
                                steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Approximate (G Gᵀ)^−½ G using fixed 5-step Newton–Schulz."""
    a, b, c = 3.4445, -4.7750, 2.0315            # Muon coefficients
    X = G if G.size(0) <= G.size(1) else G.t()   # ensure cols ≥ rows
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.t()        # (rows × rows)
        X = a * X + (b * A + c * A @ A) @ X
    return X if G.size(0) <= G.size(1) else X.t()
# --------------------------------------------------------------------- #


class local_nsmuon(Optimizer):
    """Muon optimizer with shard-wise updates inside DataParallel."""

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 momentum: float = 0.95,
                 ns_steps: int = 5,
                 betas: tuple = (0.95, 0.95),    # for AdamW (1-D tensors)
                 eps: float = 1e-8,
                 weight_decay: float = 0.1,
                 dp_group: Optional[dist.ProcessGroup] = None):

        defaults = dict(lr=lr,
                        momentum=momentum,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.ns_steps = ns_steps
        self.dp_group = dp_group
        self.world = dist.get_world_size(dp_group)
        self.rank = dist.get_rank(dp_group)

        # ------------------------------------------------------------------
        # state allocation:  matrix → momentum shard (shape rows × cols/w),
        #                    vector → AdamW buffers (full length, replicated)
        # ------------------------------------------------------------------
        for p in self.param_groups[0]["params"]:
            if p.ndim >= 2:
                local_cols = (p.shape[1] + self.world - 1) // self.world
                shape = (p.shape[0], local_cols)
                self.state[p] = {"momentum": torch.zeros(shape,
                                                         dtype=p.dtype,
                                                         device=p.device)}
            else:
                # lazy-init state for 1-D params (filled on first step)
                self.state[p] = {}

    # ------------- helpers ------------------------------------------------
    def _col_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a view of columns belonging to this rank."""
        return tensor[:, self.rank::self.world]

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def step(self,
             closure=None):                                      # noqa: C901
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

                # ------------ 0. global gradient mean (DDP) ---------------
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM,
                                group=self.dp_group)
                p.grad.div_(self.world)

                if p.ndim >= 2:
                    # ======================================================
                    # MATRIX PARAM  →  MUON shard update
                    # ======================================================
                    state = self.state[p]
                    if not state:
                        local_cols = (p.shape[1] + self.world - 1) // self.world
                        shape = (p.shape[0], local_cols)
                        self.state[p] = {"momentum": torch.zeros(shape,
                                                         dtype=p.dtype,
                                                         device=p.device)}
                    g_shard = self._col_slice(p.grad).contiguous()   # ensure contiguous
                    mom = self.state[p]["momentum"]
                    mom.mul_(mu).add_(g_shard)

                    O = zeropower_via_newtonschulz5(mom, self.ns_steps)
                    # scale to match AdamW update RMS (~0.2 · √max(A,B))
                    scale = 0.2 * math.sqrt(max(p.shape))
                    delta = -lr * scale * O

                    # weight decay then add update *on shard only*
                    w_shard = self._col_slice(p.data)
                    w_shard.mul_(1 - lr * wd)
                    w_shard.add_(delta)

                    # -------- gather shards on rank-0 --------------------
                    if self.rank == 0:
                        bucket: List[torch.Tensor] = [
                            torch.empty_like(w_shard) for _ in range(self.world)
                        ]
                    else:
                        bucket = None

                    dist.gather(w_shard.contiguous(),  # contiguous for NCCL
                                gather_list=bucket,
                                dst=0,
                                group=self.dp_group)

                    # -------- rank-0 stitches full weight ---------------
                    if self.rank == 0:
                        full = torch.empty_like(p.data)
                        for i, shard in enumerate(bucket):
                            full[:, i::self.world] = shard
                    else:
                        full = torch.empty_like(p.data)

                    # -------- broadcast full weight to all ranks ---------
                    dist.broadcast(full, src=0, group=self.dp_group)
                    p.data.copy_(full)

                else:
                    # ======================================================
                    # VECTOR PARAM  →  AdamW (standard)
                    # ======================================================
                    state = self.state[p]
                    if not state:                       # lazy initialisation
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


# -------------------------- quick usage ---------------------------------
# >>> dist.init_process_group("nccl")
# >>> model = MyTransformer().cuda()
# >>> model = torch.nn.parallel.DistributedDataParallel(model)
# >>> optim = local_nsmuon(model.parameters())
# >>> for batch in loader:
# ...     loss = model(batch).loss
# ...     loss.backward()
# ...     optim.step()
# ...     optim.zero_grad()
# -----------------------------------------------------------------------






