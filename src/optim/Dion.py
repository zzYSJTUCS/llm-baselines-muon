import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist


class Dion(Optimizer):
    """
    Centralized Dion optimizer for low-rank momentum-based updates.

    Implements Algorithm 1: Centralized Dion as a PyTorch Optimizer.

    Each parameter tensor X maintains:
      - M: momentum matrix of same shape as X
      - Q: right factor for low-rank approximation (shape: [n, rank])

    Steps per iteration:
      1. Compute gradient G for X
      2. B = M + G
      3. (P, R) = power_iteration(B, Q)
      4. Error feedback: M = B - (1 - mu) * P @ R.T
      5. ColumNormalize columns of R -> Q
      6. Update parameter: X -= lr * P @ Q.T
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        mu: float = 0.2,
        rank_factor: int = 0.5,
        orthogonalize: bool = True,
        eps: float = 1e-8,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.99,
    ):
        """
        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate (eta)
            mu: Error feedbcak decay coefficient
            rank: target rank for low-rank approximation
            orthogonalize: whether to orthogonalize P via QR decomposition
        """
        defaults = dict(lr=lr, mu=mu, rank_factor=rank_factor, orthogonalize=orthogonalize, eps = 1e-8, adam_beta1 = 0.9,
        adam_beta2 = 0.999)
        super().__init__(params, defaults)

        # Initialize state for each parameter tensor
        for group in self.param_groups:
            for p in group['params']:
                # momentum matrix M
                self.state[p]['M'] = torch.zeros_like(p)
                # random Q factor: shape [n, rank]
                if p.ndim == 1:
                    # Adam state
                    self.state['step'] = 0
                    self.state['exp_avg'] = torch.zeros_like(p)
                    self.state['exp_avg_sq'] = torch.zeros_like(p)
                else:
                    n = p.shape[-1]
                    m = p.numel()/n
                    rank = int(group['rank_factor'] * min(m, n))
                    self.state[p]['Q'] = torch.randn(n, max(rank,1), device=p.device)

    @staticmethod
    def _column_normalize(R: torch.Tensor) -> torch.Tensor:
        """Normalize each column of R by its L2 norm."""
        norms = R.norm(p=2, dim=0, keepdim=True)
        return R.div_(norms.clamp(min=1e-8))

    @staticmethod
    def _power_iteration(
        B: torch.Tensor,
        Q: torch.Tensor,
        orthogonalize: bool = True
    ) -> (torch.Tensor, torch.Tensor):
        """
        Low-rank approximation via one power iteration step.

        Args:
            B: input matrix [m, n]
            Q: current right factor [n, rank]
            orthogonalize: whether to QR-decompose P

        Returns:
            P: orthogonal factor [m, rank]
            R: factor [n, rank]
        """
        # Multiply: P = B @ Q
        P = B.matmul(Q)
        # Orthogonalize P if desired
        if orthogonalize:
            P, _ = torch.linalg.qr(P, mode='reduced')
        # Compute R = B^T @ P
        R = B.T.matmul(P)
        return P, R

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns:
            loss (if closure provided)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Loop over parameter groups
        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            orthogonalize = group['orthogonalize']
            eps = group['eps']
            beta1 = group['adam_beta1']
            beta2 = group['adam_beta2']
            for p in group['params']:
                if p.grad is None:
                    continue
                if dist.is_initialized():
                    p.grad.div(dist.get_world_size())
                state = self.state[p]
                if p.ndim == 1:
                    # Use Adam for 1D params
                    if 'exp_avg' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']
                    exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    # Access state
                else:
                    state = self.state[p]
                    M = state['M']
                    Q = state['Q']
                    G = p.grad

                    # Compute B = M + G
                    B = M + G

                    # Low-rank power iteration
                    P, R = self._power_iteration(B, Q, orthogonalize)
                    
                    # Error feedback : M = B - (1 - mu) * (P @ R^T)
                    M.copy_(B.sub(P.matmul(R.T).mul(1 - mu)))

                    # Column-normalize R to form new Q
                    Q.copy_(self._column_normalize(R))

                    # Parameter update: X -= lr * P @ Q^T
                    update = P.matmul(Q.T).mul(lr)
                    p.mul(1 - lr * 0.1) #wright decay
                    p.add_(-update)

        return loss


