"""
Build an event-transition probability matrix from decoded GENERATED sequences.

Reads chunked .npy files produced by ``Generator.generate_for_matrix`` (each
chunk has shape ``(B, seq_len, F)`` with at least feature index 0 = event type)
and produces matrices with the **same row/column ordering** as the cached
original at ``cache/transition_matrix/``, so the two are directly comparable
element-by-element.

Reuses helpers from ``create_transition_matrix.py`` to guarantee identical
denormalization, identical event-type set, and identical matrix layout.

Each generated sequence (length 50) is treated as a "match" so transitions are
NOT counted across sequence boundaries.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tokenizer.config import event_ids, event_types_mapping  # noqa: F401  (used via helper)

from diffusion_transformer.evaluation.create_transition_matrix import (
    get_non_ignored_event_types,
    _count_transitions_in_match,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_transition_matrix_generated(input_dir, output_dir):
    """
    Build transition matrix from chunked event-type .npy files.

    Args:
        input_dir: Directory containing ``batch_*.npy`` chunks of shape
            ``(B, seq_len, F)`` (only column 0 is read).
        output_dir: Directory to save the resulting matrices into. Filenames
            mirror those produced by ``create_transition_matrix.py``.

    Returns:
        Tuple of (prob_df, count_df, mapping).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(input_path.glob('batch_*.npy'))
    if not chunk_files:
        raise FileNotFoundError(f"No batch_*.npy files found in {input_path}")

    logger.info(f"Found {len(chunk_files)} chunk files in {input_path}")

    non_ignored_events = get_non_ignored_event_types()
    all_event_ids = sorted(event_ids.values())          # for denormalization
    matrix_event_ids = sorted(non_ignored_events.keys())  # rows/cols of matrix

    n_types = len(matrix_event_ids)
    count_matrix = np.zeros((n_types, n_types), dtype=np.int64)
    id_to_idx = {event_id: idx for idx, event_id in enumerate(matrix_event_ids)}
    event_names = [non_ignored_events[eid] for eid in matrix_event_ids]

    total_sequences = 0
    total_transitions = 0

    for chunk_path in chunk_files:
        chunk = np.load(chunk_path)  # (B, seq_len, F)
        if chunk.ndim != 3:
            raise ValueError(
                f"Expected 3-D chunk (B, seq_len, F) at {chunk_path}, got shape {chunk.shape}"
            )

        for seq_idx in range(chunk.shape[0]):
            sequence = chunk[seq_idx]  # (seq_len, F)
            total_transitions += _count_transitions_in_match(
                sequence, all_event_ids, id_to_idx, count_matrix
            )
            total_sequences += 1

        logger.debug(f"  Processed {chunk_path.name}: {chunk.shape[0]} sequences")

    logger.info(f"Processed {total_sequences} sequences, "
                f"counted {total_transitions} transitions")

    # ---- normalize to probabilities --------------------------------------
    prob_matrix = count_matrix.astype(float)
    for i in range(n_types):
        row_sum = prob_matrix[i].sum()
        if row_sum > 0:
            prob_matrix[i] /= row_sum
        else:
            prob_matrix[i] = 1.0 / n_types

    # ---- save outputs (mirroring original filenames) ----------------------
    np.save(output_path / 'event_transition_probabilities.npy', prob_matrix)
    np.save(output_path / 'event_transition_counts.npy', count_matrix)
    logger.info("Saved NumPy arrays")

    prob_df = pd.DataFrame(prob_matrix, index=event_names, columns=event_names)
    count_df = pd.DataFrame(count_matrix, index=event_names, columns=event_names)
    prob_df.to_pickle(output_path / 'event_transition_probabilities.pkl')
    count_df.to_pickle(output_path / 'event_transition_counts.pkl')
    logger.info("Saved pandas pickles")

    prob_df.to_csv(output_path / 'event_transition_probabilities_original.csv')
    count_df.to_csv(output_path / 'event_transition_counts_original.csv')
    logger.info("Saved CSVs")

    mapping = {
        'id_to_index': id_to_idx,
        'index_to_id': {idx: eid for eid, idx in id_to_idx.items()},
        'index_to_name': {idx: non_ignored_events[eid] for idx, eid in enumerate(matrix_event_ids)},
        'event_type_ids': matrix_event_ids,
    }
    with open(output_path / 'event_type_mapping.pkl', 'wb') as f:
        pickle.dump(mapping, f)
    logger.info("Saved event type mapping")

    logger.info(f"\nMatrix shape: {prob_matrix.shape}")
    logger.info(f"Row sums: min={prob_matrix.sum(axis=1).min():.6f}, "
                f"max={prob_matrix.sum(axis=1).max():.6f}")

    logger.info("\nTop 5 transitions in generated matrix:")
    flat_indices = np.argsort(prob_matrix.ravel())[::-1][:5]
    for idx in flat_indices:
        i, j = idx // n_types, idx % n_types
        logger.info(f"  {event_names[i]:20s} -> {event_names[j]:20s}: {prob_matrix[i, j]:.4f}")

    logger.info("Complete.")
    return prob_df, count_df, mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build transition matrix from generated event-type chunks."
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory containing batch_*.npy chunks '
             '(typically <output_dir>/event_types from generate_for_matrix).'
    )
    parser.add_argument(
        '--output_dir', type=str, default='cache/transition_matrix_generated',
        help='Where to write the resulting matrix files.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    create_transition_matrix_generated(input_dir, output_dir)


if __name__ == '__main__':
    main()
