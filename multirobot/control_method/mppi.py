import torch
from pytorch_mppi import MPPI


class MPPIController(MPPI):
    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=1000, horizon=15, device="cpu",
                 terminal_state_cost=None, lambda_=1., noise_mu=None, u_min=None, u_max=None, max_action_norm=None):
        """
        Extended MPPI controller with optional L2 norm constraint on actions.

        Args:
            max_action_norm: Maximum L2 norm allowed for control actions.
                            Set to None to disable norm constraints.
            Other parameters: Same as base MPPI class.
        """
        super().__init__(
            dynamics=dynamics,
            running_cost=running_cost,
            nx=nx,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon,
            device=device,
            terminal_state_cost=terminal_state_cost,
            lambda_=lambda_,
            noise_mu=noise_mu,
            u_min=u_min,
            u_max=u_max
        )
        self.max_action_norm = max_action_norm

    def _compute_perturbed_action_and_noise(self):
        """Generate and constrain perturbation noise with L2 norm constraint"""
        # Sample noise from Gaussian distribution
        noise = self.noise_dist.rsample((self.K, self.T))

        # Apply L2 norm constraint if specified
        if self.max_action_norm is not None:
            noise_norm = torch.norm(noise, dim=-1, keepdim=True)
            noise = noise / (noise_norm + 1e-6) * self.max_action_norm

        # Generate perturbed actions by adding noise to nominal control sequence
        perturbed_action = self.U + noise
        perturbed_action = self._sample_specific_actions(perturbed_action)

        # Apply bounds (including potential norm constraint)
        self.perturbed_action = self._bound_action(perturbed_action)

        # Store effective noise after all constraints
        self.noise = self.perturbed_action - self.U

    def _bound_action(self, action):
        """Apply all action constraints (box + norm)"""
        # First apply original box constraints (u_min/u_max)
        action = super()._bound_action(action)

        # Apply additional norm constraint if enabled
        if self.max_action_norm is not None:
            action_norm = torch.norm(action, dim=-1, keepdim=True)
            scaling = torch.minimum(
                torch.ones_like(action_norm),
                self.max_action_norm / (action_norm + 1e-6)
            )
            action = action * scaling

        return action