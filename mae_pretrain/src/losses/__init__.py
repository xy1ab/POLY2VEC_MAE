"""Loss package for MAE pretraining."""

from .recon_mag_phase import compute_mag_phase_losses

__all__ = ["compute_mag_phase_losses"]
