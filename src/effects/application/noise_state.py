"""The noise on ``e``: Gaussian, scaled to the spread of ``e`` itself (FR-056)."""

from __future__ import annotations

import torch

#: Running-covariance decay for the noise on ``e`` (FR-056).
NOISE_DECAY = 0.99

#: Steps between two factorizations of the running covariance: a GPU
#: eigendecomposition stalls the step on the host. ``Σ`` still updates every
#: step.
NOISE_REFACTOR_EVERY = 10


class NoiseState:
    """Gaussian noise on ``e`` scaled to the spread of ``e`` itself (FR-056).

    The noise has covariance ``r² Σ``: ``Σ`` a running covariance of the
    batches' ``e`` with the gradient stopped, ``r`` rising linearly from 0 to
    the ratio over the first ``ramp_steps`` optimizer steps.

    Update and draw run outside autocast and in float64: a bf16 covariance is
    neither symmetric nor positive semi-definite, and its factorization fails
    within the first steps.
    """

    def __init__(
        self, ratio: float, ramp_steps: int, *,
        decay: float = NOISE_DECAY,
        refactor_every: int = NOISE_REFACTOR_EVERY,
    ) -> None:
        self.ratio = ratio
        self.ramp_steps = max(int(ramp_steps), 1)
        self.decay = decay
        self.sigma: torch.Tensor | None = None
        self.refactor_every = max(int(refactor_every), 1)
        #: ``V √Λ`` of the last factorized ``Σ`` and the draw it was taken at.
        self._factor: torch.Tensor | None = None
        self._draws_since_factor = 0

    def state_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "factor": self._factor, "draws_since_factor": self._draws_since_factor,
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        """Restore :meth:`state_dict`'s keys and ignore any other, so a resume
        file still holding the ``step`` the state once saved continues."""
        def on(tensor):
            return tensor.to(device) if tensor is not None else None

        self.sigma = on(state["sigma"])
        self._factor = on(state["factor"])
        self._draws_since_factor = state["draws_since_factor"]

    def scale(self, step: int) -> float:
        """``r`` at training step ``step``."""
        return self.ratio * min(1.0, step / self.ramp_steps)

    def update(self, matrix: torch.Tensor) -> None:
        """Fold one batch's covariance into ``Σ``."""
        if matrix.shape[0] < 2:
            return
        with torch.no_grad(), _exact(matrix):
            vectors = matrix.detach().double()
            centred = vectors - vectors.mean(dim=0, keepdim=True)
            covariance = centred.T @ centred / (vectors.shape[0] - 1)
            covariance = (covariance + covariance.T) / 2
            if self.sigma is None:
                self.sigma = covariance
            else:
                self.sigma = (
                    self.decay * self.sigma + (1.0 - self.decay) * covariance
                )

    def apply(self, matrix: torch.Tensor | None, *, step: int) -> torch.Tensor | None:
        """``matrix`` plus ``L z``, where ``L Lᵀ = r²Σ``, at training step
        ``step``; a batch with no ``e`` stays None.

        ``L = V √Λ`` with eigenvalues clamped at zero, so a singular ``Σ``
        (a batch narrower than ``e_dim``) never fails to factorize.
        """
        if matrix is None:
            return None
        self.update(matrix)
        r = self.scale(step)
        if r <= 0.0 or self.sigma is None:
            return matrix
        with torch.no_grad(), _exact(matrix):
            if self._factor is None or self._draws_since_factor >= self.refactor_every:
                values, vectors = torch.linalg.eigh(self.sigma)
                self._factor = vectors * values.clamp(min=0.0).sqrt()
                self._draws_since_factor = 0
            self._draws_since_factor += 1
            factor = r * self._factor
            draw = torch.randn(
                matrix.shape[0], factor.shape[0],
                device=matrix.device, dtype=factor.dtype,
            ) @ factor.T
        return matrix + draw.to(matrix.dtype)


def _exact(matrix: torch.Tensor):
    """A region where autocast does not lower the noise arithmetic's precision."""
    return torch.autocast(device_type=matrix.device.type, enabled=False)
