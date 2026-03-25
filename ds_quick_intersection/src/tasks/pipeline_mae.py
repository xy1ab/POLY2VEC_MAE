"""Downstream MAE reconstruction task wrapper.

This wrapper intentionally imports only the public pretrain pipeline API, which
keeps downstream decoupled from pretrain internal module structure.
"""

from __future__ import annotations

from engine.pipeline import PolyMaeReconstructionPipeline


class MaeReconstructionPipeline(PolyMaeReconstructionPipeline):
    """Compatibility wrapper around pretrain public reconstruction pipeline."""

    pass
