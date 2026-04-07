"""Contract tests for the EMA vector quantizer."""

from __future__ import annotations

import torch
import unittest

from ..src.models.quantizer import EMAVectorQuantizer


class QuantizerContractTest(unittest.TestCase):
    """Shape and initialization contracts for the EMA quantizer."""

    def test_quantizer_forward_returns_expected_shapes(self) -> None:
        """Quantizer should preserve latent-grid shape and emit 2D index grids."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=16,
            embedding_dim=8,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        latents = torch.randn(2, 8, 4, 3)

        outputs = quantizer(latents)

        self.assertEqual(outputs.quantized.shape, latents.shape)
        self.assertEqual(outputs.indices.shape, (2, 4, 3))
        self.assertEqual(outputs.vq_loss.ndim, 0)

    def test_lookup_indices_restores_quantized_grid_shape(self) -> None:
        """Lookup path should map `[B,H,W]` indices back to `[B,C,H,W]` tensors."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        latents = torch.randn(1, 4, 2, 2)
        outputs = quantizer(latents)

        restored = quantizer.lookup_indices(outputs.indices)

        self.assertEqual(restored.shape, latents.shape)

    def test_codebook_initialization_from_vectors_marks_quantizer_ready(self) -> None:
        """Explicit codebook initialization should flip the initialized flag."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        vectors = torch.randn(32, 4)

        quantizer.initialize_codebook(vectors, num_iters=2)

        self.assertTrue(quantizer.is_initialized)
        self.assertEqual(quantizer.codebook.shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
