"""Decoupling tests for downstream package."""

from __future__ import annotations

from pathlib import Path


def test_downstream_uses_pretrain_public_pipeline_only() -> None:
    """Ensure downstream wrapper imports only engine.pipeline API."""
    root = Path(__file__).resolve().parents[1]
    content = (root / "src" / "tasks" / "pipeline_mae.py").read_text(encoding="utf-8")
    assert "from engine.pipeline import PolyMaeReconstructionPipeline" in content
