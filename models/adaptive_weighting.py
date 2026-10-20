# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class AdaptiveLossBalancer(nn.Module):
    """
    V2: bounded gradient-based adaptive balancing for
        physics / SSIM / MSE.

    Design goals:
    1) Do NOT use 1/raw_loss initialization.
    2) Do NOT introduce learnable log-variance parameters.
    3) Keep all weights positive, bounded and normalized so sum(weights)=3.
    4) Update weights from gradient norms on a shared output-layer parameter,
       with a relative training-rate correction and EMA smoothing.

    This is a deliberately conservative GradNorm-style controller, not a
    trainable sub-network.  It cannot drive the total loss negative and it
    cannot produce weights such as 1e5.
    """

    NAMES = ('physics', 'ssim', 'mse')
    VERSION = 'bounded_grad_balance_v2'

    def __init__(self, config):
        super().__init__()
        self.eps = float(getattr(config, 'loss_balance_eps', 1e-12))
        self.update_interval = int(
            getattr(config, 'loss_balance_update_interval', 10)
        )
        self.ema = float(getattr(config, 'loss_balance_ema', 0.90))
        self.alpha = float(getattr(config, 'loss_balance_alpha', 0.50))
        self.min_weight = float(
            getattr(config, 'loss_balance_min_weight', 1e-3)
        )
        self.max_weight = float(
            getattr(config, 'loss_balance_max_weight', 3.0)
        )

        if self.update_interval < 1:
            raise ValueError('loss_balance_update_interval must be >= 1')
        if not (0.0 <= self.ema < 1.0):
            raise ValueError('loss_balance_ema must be in [0, 1)')
        if self.alpha < 0.0:
            raise ValueError('loss_balance_alpha must be >= 0')
        if self.min_weight <= 0.0:
            raise ValueError('loss_balance_min_weight must be > 0')
        if self.max_weight <= self.min_weight:
            raise ValueError('loss_balance_max_weight must exceed min_weight')

        n = len(self.NAMES)
        self.register_buffer('weights', torch.ones(n, dtype=torch.float32))
        self.register_buffer('initial_losses', torch.ones(n, dtype=torch.float32))
        self.register_buffer('grad_ema', torch.ones(n, dtype=torch.float32))
        self.register_buffer('step_counter', torch.zeros((), dtype=torch.long))
        self.register_buffer('initialized_flag', torch.zeros((), dtype=torch.uint8))

    @property
    def initialized(self):
        return bool(self.initialized_flag.item())

    @torch.no_grad()
    def initialize_from_means(self, loss_means):
        vals = []
        for name in self.NAMES:
            value = max(float(loss_means[name]), self.eps)
            vals.append(value)

        self.initial_losses.copy_(
            torch.tensor(vals, dtype=self.initial_losses.dtype,
                         device=self.initial_losses.device)
        )
        # IMPORTANT: start from equal weights, not 1/loss.
        self.weights.fill_(1.0)
        self.grad_ema.fill_(1.0)
        self.step_counter.zero_()
        self.initialized_flag.fill_(1)

    @staticmethod
    def _grad_norm(loss, shared_parameter):
        grad = torch.autograd.grad(
            loss,
            shared_parameter,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        if grad is None:
            return torch.zeros((), device=loss.device, dtype=loss.dtype)
        return torch.linalg.vector_norm(grad.detach(), ord=2)

    @torch.no_grad()
    def _project_weights(self, w):
        """Positive bounded weights with mean exactly 1 (sum=N)."""
        n = float(len(self.NAMES))
        w = torch.nan_to_num(
            w,
            nan=1.0,
            posinf=self.max_weight,
            neginf=self.min_weight
        )
        w = torch.clamp(w, min=self.min_weight, max=self.max_weight)
        w = w * (n / (w.sum() + self.eps))

        # One more clamp/renormalize pass after normalization.
        w = torch.clamp(w, min=self.min_weight, max=self.max_weight)
        w = w * (n / (w.sum() + self.eps))
        return w

    def maybe_update(self, components, shared_parameter):
        """
        Update only every `update_interval` optimization steps.

        Desired weight is based on:
          target_grad_i / raw_grad_i

        target_grad_i also contains a relative-training-rate term so a loss
        that is decreasing more slowly receives somewhat more attention.
        The update is heavily smoothed and strictly bounded.
        """
        if not self.initialized:
            raise RuntimeError('AdaptiveLossBalancer has not been initialized.')

        step = int(self.step_counter.item())
        self.step_counter.add_(1)

        if step % self.update_interval != 0:
            return self.current_info()

        grad_vals = []
        for name in self.NAMES:
            g = self._grad_norm(components[name], shared_parameter)
            grad_vals.append(g)

        grads = torch.stack(grad_vals).to(self.weights.device)
        grads = torch.nan_to_num(grads, nan=0.0, posinf=1e12, neginf=0.0)

        # Update a smooth raw-gradient estimate.
        if step == 0:
            new_grad_ema = torch.clamp(grads, min=self.eps)
        else:
            new_grad_ema = (
                self.ema * self.grad_ema +
                (1.0 - self.ema) * torch.clamp(grads, min=self.eps)
            )

        with torch.no_grad():
            self.grad_ema.copy_(new_grad_ema)

            current_losses = torch.stack([
                components[name].detach().to(self.weights.device)
                for name in self.NAMES
            ])
            current_losses = torch.clamp(current_losses, min=self.eps)

            # Relative inverse training rate (initial value ~= 1 for all).
            loss_ratio = current_losses / torch.clamp(
                self.initial_losses, min=self.eps
            )
            relative_rate = loss_ratio / (loss_ratio.mean() + self.eps)

            g = torch.clamp(self.grad_ema, min=self.eps)
            mean_g = g.mean()
            target_g = mean_g * torch.pow(
                torch.clamp(relative_rate, min=self.eps),
                self.alpha
            )

            desired = target_g / g
            desired = self._project_weights(desired)

            # Conservative EMA update of the weights themselves.
            blended = self.ema * self.weights + (1.0 - self.ema) * desired
            blended = self._project_weights(blended)
            self.weights.copy_(blended)

        return self.current_info()

    def forward(self, components):
        if not self.initialized:
            raise RuntimeError('AdaptiveLossBalancer has not been initialized.')

        # weights are buffers, i.e. constants for model back-propagation.
        weighted = []
        for i, name in enumerate(self.NAMES):
            weighted.append(self.weights[i].detach() * components[name])

        main_total = torch.stack(weighted).mean()
        return main_total, self.current_info()

    @torch.no_grad()
    def current_weights(self):
        return {
            name: float(self.weights[i].item())
            for i, name in enumerate(self.NAMES)
        }

    @torch.no_grad()
    def current_info(self):
        return {
            'weights': self.current_weights(),
            'grad_ema': {
                name: float(self.grad_ema[i].item())
                for i, name in enumerate(self.NAMES)
            },
            'step': int(self.step_counter.item()),
            'version': self.VERSION,
        }
