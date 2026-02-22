"""
Diagnostic script: Autoencoder encode-decode timestamp fidelity test.

Tests whether the autoencoder's encode→decode cycle introduces non-monotonic
timestamps, isolating the decoding step from the diffusion process.

Approach:
    1. Load 5 real sequences of 50 events from cached match data.
    2. Evaluate the original sequences with SequenceEvaluator (sanity check).
    3. Encode sequences via the autoencoder encoder, then decode them back.
    4. Evaluate the reconstructed sequences with the same evaluator.
    5. Compare per-event timestamp drift between original and reconstructed.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Ensure project root is on sys.path so imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_transformer.data.event_autoencoder_model import EventAutoencoder
from diffusion_transformer.evaluation.evaluator import SequenceEvaluator
from diffusion_transformer.utils import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SEQUENCES = 5
SEQUENCE_LENGTH = 50
AUTOENCODER_PATH = PROJECT_ROOT / "diffusion_transformer" / "models" / "autoencoder.pt"
CACHE_DIR = PROJECT_ROOT / "cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sequences_from_cache(
    cache_dir: Path,
    num_sequences: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load contiguous event sub-sequences and their matching embeddings.

    Uses the same cached .npy files that the training pipeline produces,
    guaranteeing consistency with the rest of the codebase.

    Args:
        cache_dir: Path to the cache directory.
        num_sequences: Number of sequences to extract.
        sequence_length: Number of events per sequence.

    Returns:
        Tuple of (events, embeddings) arrays each shaped
        (num_sequences, sequence_length, feature_dim).
    """
    event_files = sorted(cache_dir.glob("*_events.npy"))
    embedding_files_map = {}
    for ef in sorted(cache_dir.glob("*_embeddings_full.npy")):
        match_id = ef.stem.replace("_embeddings_full", "")
        embedding_files_map[match_id] = ef

    events_list: list[np.ndarray] = []
    embeddings_list: list[np.ndarray] = []

    for event_file in event_files:
        if len(events_list) >= num_sequences:
            break

        match_id = event_file.stem.replace("_events", "")
        if match_id not in embedding_files_map:
            continue

        match_events = np.load(event_file, allow_pickle=True)
        match_embeddings = np.load(embedding_files_map[match_id])

        if len(match_events) < sequence_length:
            continue

        # Take the first contiguous subsequence of the required length
        events_list.append(match_events[:sequence_length])
        embeddings_list.append(match_embeddings[:sequence_length])

    if len(events_list) < num_sequences:
        raise RuntimeError(
            f"Could only find {len(events_list)} valid matches "
            f"(needed {num_sequences} with >= {sequence_length} events)."
        )

    events_batch = np.stack(events_list[:num_sequences])
    embeddings_batch = np.stack(embeddings_list[:num_sequences])

    logger.info(
        "Loaded %d sequences — events shape: %s, embeddings shape: %s",
        num_sequences,
        events_batch.shape,
        embeddings_batch.shape,
    )
    return events_batch, embeddings_batch


def load_autoencoder(model_path: Path, input_dim: int, device: str) -> EventAutoencoder:
    """
    Load the pre-trained autoencoder.

    Args:
        model_path: Path to saved state_dict.
        input_dim: Dimensionality of the event feature vector.
        device: Torch device string.

    Returns:
        Loaded EventAutoencoder in eval mode.
    """
    model = EventAutoencoder(input_dim=input_dim, latent_dim=32)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Autoencoder loaded from %s (input_dim=%d)", model_path, input_dim)
    return model


def encode_decode(
    model: EventAutoencoder,
    events: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Run events through encoder → decoder and return reconstructed events.

    Args:
        model: Loaded autoencoder.
        events: Array of shape (batch, seq_len, feature_dim).
        device: Torch device string.

    Returns:
        Reconstructed events with the same shape as input.
    """
    batch, seq_len, feat_dim = events.shape
    flat = events.reshape(-1, feat_dim)
    tensor = torch.tensor(flat, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed, _ = model(tensor)

    return reconstructed.cpu().numpy().reshape(batch, seq_len, feat_dim)


def compare_timestamps(
    evaluator: SequenceEvaluator,
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> None:
    """
    Print a per-event timestamp comparison for the first sequence.

    Args:
        evaluator: Initialised SequenceEvaluator (used for denormalization).
        original: Original event batch (N, seq_len, feat_dim).
        reconstructed: Reconstructed event batch (N, seq_len, feat_dim).
    """
    logger.info("=" * 70)
    logger.info("PER-EVENT TIMESTAMP COMPARISON (first sequence)")
    logger.info("=" * 70)
    logger.info(
        "%4s | %22s | %22s | %8s",
        "Idx", "Original (P MM:SS tot)", "Reconstructed (P MM:SS tot)", "Δ(sec)",
    )
    logger.info("-" * 70)

    for i in range(original.shape[1]):
        o_tot, o_p, o_m, o_s = evaluator._extract_timestamp(original[0, i])
        r_tot, r_p, r_m, r_s = evaluator._extract_timestamp(reconstructed[0, i])
        delta = r_tot - o_tot
        logger.info(
            "%4d | P%d %02d:%02d %6ds | P%d %02d:%02d %6ds | %+6ds",
            i, o_p, o_m, o_s, o_tot, r_p, r_m, r_s, r_tot, delta,
        )


def main() -> None:
    """Run the end-to-end autoencoder timestamp fidelity diagnostic."""
    setup_logging(level="INFO")
    logger.info("=" * 70)
    logger.info("AUTOENCODER TIMESTAMP FIDELITY DIAGNOSTIC")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    events, embeddings = load_sequences_from_cache(
        CACHE_DIR, NUM_SEQUENCES, SEQUENCE_LENGTH
    )
    input_dim = events.shape[-1]

    # ------------------------------------------------------------------
    # 2. Evaluate ORIGINAL sequences (sanity check — should pass)
    # ------------------------------------------------------------------
    evaluator = SequenceEvaluator(
        cache_dir=str(CACHE_DIR),
        transition_tolerance=0.0,
        time_tolerance=0,
    )

    logger.info("\n>>> Evaluating ORIGINAL sequences <<<")
    original_results = evaluator.evaluate_batch(events)

    # ------------------------------------------------------------------
    # 3. Encode → Decode
    # ------------------------------------------------------------------
    autoencoder = load_autoencoder(AUTOENCODER_PATH, input_dim, DEVICE)
    reconstructed = encode_decode(autoencoder, events, DEVICE)

    # ------------------------------------------------------------------
    # 4. Evaluate RECONSTRUCTED sequences
    # ------------------------------------------------------------------
    logger.info("\n>>> Evaluating RECONSTRUCTED (encode→decode) sequences <<<")
    reconstructed_results = evaluator.evaluate_batch(reconstructed)

    # ------------------------------------------------------------------
    # 5. Per-event timestamp comparison
    # ------------------------------------------------------------------
    compare_timestamps(evaluator, events, reconstructed)

    # ------------------------------------------------------------------
    # 6. Reconstruction error summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RECONSTRUCTION ERROR SUMMARY")
    logger.info("=" * 70)

    period_idx = evaluator.period_idx
    second_idx = evaluator.second_idx
    minute_idx = evaluator.minute_idx

    for label, idx in [("period", period_idx), ("second", second_idx), ("minute", minute_idx)]:
        orig_vals = events[:, :, idx]
        recon_vals = reconstructed[:, :, idx]
        mae = np.mean(np.abs(orig_vals - recon_vals))
        max_err = np.max(np.abs(orig_vals - recon_vals))
        logger.info(
            "  %-8s  MAE=%.6f  MaxErr=%.6f  (normalised scale)",
            label, mae, max_err,
        )

    overall_mae = np.mean(np.abs(events - reconstructed))
    logger.info("  %-8s  MAE=%.6f", "overall", overall_mae)

    # Show raw normalised value pairs for the first sequence's minute feature
    logger.info("\n  Minute (idx %d) raw normalised values — first sequence:", minute_idx)
    for i in range(min(10, events.shape[1])):
        o_val = events[0, i, minute_idx]
        r_val = reconstructed[0, i, minute_idx]
        o_min = round(o_val * 60)
        r_min = round(r_val * 60)
        logger.info(
            "    event %2d: orig=%.6f (→%2d min)  recon=%.6f (→%2d min)  Δ=%.6f",
            i, o_val, o_min, r_val, r_min, r_val - o_val,
        )

    # ------------------------------------------------------------------
    # 7. Final verdict
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)
    logger.info(
        "Original sequences  — pass rate: %.2f%% (%d/%d)",
        original_results["pass_rate"] * 100,
        original_results["passed_sequences"],
        original_results["total_sequences"],
    )
    logger.info(
        "  Transition violations: %d, Time violations: %d",
        original_results["total_transition_violations"],
        original_results["total_time_violations"],
    )
    logger.info(
        "Reconstructed sequences — pass rate: %.2f%% (%d/%d)",
        reconstructed_results["pass_rate"] * 100,
        reconstructed_results["passed_sequences"],
        reconstructed_results["total_sequences"],
    )
    logger.info(
        "  Transition violations: %d, Time violations: %d",
        reconstructed_results["total_transition_violations"],
        reconstructed_results["total_time_violations"],
    )

    if reconstructed_results["total_time_violations"] > 0:
        logger.warning(
            "⚠  The autoencoder encode→decode cycle INTRODUCES timestamp "
            "violations. The decoder is likely a contributing factor to the "
            "non-monotonic timestamps observed in generated sequences."
        )
    else:
        logger.info(
            "✓  The autoencoder encode→decode cycle preserves timestamp "
            "monotonicity. Non-monotonic timestamps in generated sequences "
            "are likely caused by the diffusion process itself."
        )


if __name__ == "__main__":
    main()


