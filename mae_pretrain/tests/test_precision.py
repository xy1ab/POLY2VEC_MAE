"""Precision policy tests for mae_pretrain utils."""

from __future__ import annotations

from utils.precision import normalize_precision


def test_precision_alias_fp13_maps_to_fp16() -> None:
    """Ensure fp13 alias is normalized to fp16 as agreed with user."""
    assert normalize_precision("fp13") == "fp16"


def test_precision_keeps_bf16() -> None:
    """Ensure bf16 remains unchanged."""
    assert normalize_precision("bf16") == "bf16"
