"""
Event Transition Probability Matrix Generator

Reads events_df.pkl and creates a transition probability matrix showing
the probability of each event type occurring after each other event type.
Respects match boundaries and saves in multiple formats for easy loading.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tokenizer.config import event_ids, event_types_mapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_non_ignored_event_types():
    """Extract non-ignored event types from config."""
    non_ignored = {}
    for name, event_id in event_ids.items():
        if event_id in event_types_mapping and not event_types_mapping[event_id].get('ignore_event_type', False):
            non_ignored[event_id] = name
    return non_ignored


def denormalize_event_type(normalized_value, all_event_ids):
    """
    Convert normalized event type (0-1) back to original event ID.

    The tokenizer's CategoricalFeatureParser normalizes as: (index + 1) / num_categories
    where index is the position in the sorted categories list (0-based).
    So to denormalize: index = round(normalized_value * num_categories) - 1
    """
    num_categories = len(all_event_ids)
    # The tokenizer uses (index + 1) / num_categories, so we reverse that
    category_number = round(normalized_value * num_categories)  # This gives us 1 to num_categories
    original_index = category_number - 1  # Convert to 0-based index
    # Clamp to valid range
    original_index = max(0, min(original_index, num_categories - 1))
    return all_event_ids[original_index]


def _count_transitions_in_match(match_events, all_event_ids, id_to_idx, count_matrix):
    """
    Count transitions within a single match and update the count matrix.

    Args:
        match_events: NumPy array of events for one match (N, feature_dim)
        all_event_ids: Sorted list of all event IDs (for denormalization)
        id_to_idx: Dict mapping event_id to matrix index
        count_matrix: Count matrix to update (modified in place)

    Returns:
        Number of transitions counted in this match
    """
    transitions_counted = 0

    # Extract and denormalize event types (column 0)
    event_types_normalized = match_events[:, 0]
    event_type_sequence = [
        denormalize_event_type(val, all_event_ids)
        for val in event_types_normalized
    ]

    # Count transitions within this match only
    for i in range(len(event_type_sequence) - 1):
        current = event_type_sequence[i]
        next_event = event_type_sequence[i + 1]

        # Only count non-ignored event types
        if current in id_to_idx and next_event in id_to_idx:
            count_matrix[id_to_idx[current], id_to_idx[next_event]] += 1
            transitions_counted += 1

    return transitions_counted


def create_transition_matrix(events_df_path=None):
    """
    Create event transition probability matrix from events_df.pkl or individual match files.

    Args:
        events_df_path: Path to events_df.pkl file (defaults to project_root/cache/events_df.pkl)
    """
    logger.info("Starting transition matrix creation")

    # Get project root (football_events_generation directory)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Set default paths relative to project root
    if events_df_path is None:
        events_df_path = project_root / 'cache' / 'events_df.pkl'
    else:
        events_df_path = Path(events_df_path)
        if not events_df_path.is_absolute():
            events_df_path = project_root / events_df_path

    output_dir = project_root / 'cache' / 'transition_matrix'

    # Get event types from config
    non_ignored_events = get_non_ignored_event_types()
    all_event_ids = sorted(event_ids.values())  # All 34 event IDs for denormalization
    matrix_event_ids = sorted(non_ignored_events.keys())  # 29 non-ignored for matrix

    logger.info(f"Using {len(matrix_event_ids)} non-ignored event types")

    # Initialize count matrix
    n_types = len(matrix_event_ids)
    count_matrix = np.zeros((n_types, n_types), dtype=int)
    id_to_idx = {event_id: idx for idx, event_id in enumerate(matrix_event_ids)}

    # Check if events_df.pkl exists
    events_df_path_obj = Path(events_df_path)
    total_transitions = 0

    if events_df_path_obj.exists():
        # Load from events_df.pkl
        logger.info(f"Loading events from {events_df_path}")
        events_df = pd.read_pickle(events_df_path)
        logger.info(f"Loaded {len(events_df)} events from {events_df['match_id'].nunique()} matches")

        # Process each match separately (respect match boundaries)
        match_ids = events_df['match_id'].unique()

        for match_id in match_ids:
            match_df = events_df[events_df['match_id'] == match_id]
            match_events = match_df.drop(columns=['match_id']).values
            total_transitions += _count_transitions_in_match(
                match_events, all_event_ids, id_to_idx, count_matrix
            )
    else:
        # Load from individual .npy files - process each separately to maintain isolation
        logger.info("events_df.pkl not found, loading from individual match .npy files...")
        cache_path = events_df_path_obj.parent
        event_files = sorted(cache_path.glob('*_events.npy'))

        if not event_files:
            raise FileNotFoundError(f"No *_events.npy files found in {cache_path}")

        logger.info(f"Found {len(event_files)} match event files")

        for event_file in event_files:
            match_id = event_file.stem.replace('_events', '')
            match_events = np.load(event_file, allow_pickle=True)
            logger.info(f"  Processing {len(match_events)} events from match {match_id}")

            total_transitions += _count_transitions_in_match(
                match_events, all_event_ids, id_to_idx, count_matrix
            )

    logger.info(f"Recorded {total_transitions} transitions")

    # Normalize to probabilities
    prob_matrix = count_matrix.astype(float)
    for i in range(n_types):
        row_sum = prob_matrix[i].sum()
        if row_sum > 0:
            prob_matrix[i] /= row_sum
        else:
            prob_matrix[i] = 1.0 / n_types  # Uniform distribution if no transitions

    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    event_names = [non_ignored_events[eid] for eid in matrix_event_ids]

    # Save as NumPy arrays
    np.save(output_path / 'event_transition_probabilities.npy', prob_matrix)
    np.save(output_path / 'event_transition_counts.npy', count_matrix)
    logger.info("Saved NumPy arrays")

    # Save as pandas DataFrames (pickle)
    prob_df = pd.DataFrame(prob_matrix, index=event_names, columns=event_names)
    count_df = pd.DataFrame(count_matrix, index=event_names, columns=event_names)
    prob_df.to_pickle(output_path / 'event_transition_probabilities.pkl')
    count_df.to_pickle(output_path / 'event_transition_counts.pkl')
    logger.info("Saved pandas pickles")

    # Save as CSV for viewing
    prob_df.to_csv(output_path / 'event_transition_probabilities_original.csv')
    count_df.to_csv(output_path / 'event_transition_counts_original.csv')
    logger.info("Saved CSVs")

    # Save mapping
    mapping = {
        'id_to_index': id_to_idx,
        'index_to_id': {idx: eid for eid, idx in id_to_idx.items()},
        'index_to_name': {idx: non_ignored_events[eid] for idx, eid in enumerate(matrix_event_ids)},
        'event_type_ids': matrix_event_ids
    }
    with open(output_path / 'event_type_mapping.pkl', 'wb') as f:
        pickle.dump(mapping, f)
    logger.info("Saved event type mapping")

    # Print summary
    logger.info(f"\nMatrix shape: {prob_matrix.shape}")
    logger.info(f"Row sums: min={prob_matrix.sum(axis=1).min():.6f}, max={prob_matrix.sum(axis=1).max():.6f}")

    # Show top transitions
    logger.info("\nTop 5 transitions:")
    flat_indices = np.argsort(prob_matrix.ravel())[::-1][:5]
    for idx in flat_indices:
        i, j = idx // n_types, idx % n_types
        logger.info(f"  {event_names[i]:20s} → {event_names[j]:20s}: {prob_matrix[i,j]:.4f}")

    logger.info("\n✅ Complete!")
    return prob_df, count_df, mapping


def load_transition_matrix(cache_dir=None, as_dataframe=True):
    """
    Load the transition probability matrix.

    Args:
        cache_dir: Directory containing cached files (defaults to project_root/cache/transition_matrix)
        as_dataframe: If True, return pandas DataFrames; else return NumPy arrays

    Returns:
        Tuple of (prob_matrix, count_matrix, mapping)
    """
    # Get project root (football_events_generation directory)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Set default path relative to project root
    if cache_dir is None:
        cache_path = project_root / 'cache' / 'transition_matrix'
    else:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = project_root / cache_dir

    if as_dataframe:
        prob_matrix = pd.read_pickle(cache_path / 'event_transition_probabilities.pkl')
        count_matrix = pd.read_pickle(cache_path / 'event_transition_counts.pkl')
    else:
        prob_matrix = np.load(cache_path / 'event_transition_probabilities.npy')
        count_matrix = np.load(cache_path / 'event_transition_counts.npy')

    with open(cache_path / 'event_type_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    return prob_matrix, count_matrix, mapping



if __name__ == '__main__':
    # Generate matrix
    prob_df, count_df, mapping = create_transition_matrix()

