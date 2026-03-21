#!/usr/bin/env python3
"""
Script to extract and display event sequences from decoded samples.
Selects random sequences and shot-related sequences, converting event type
values to readable event names.

DECODING LOGIC (based on statsbomb_football_embeddings tokenizer):
=================================================================
The tokenizer's CategoricalFeatureParser normalizes event types using:
    normalized_value = sorted_index / num_categories

Where:
- num_categories = 34 (ALL event IDs, including 5 ignored ones)
- sorted_index = 1-indexed position in sorted(event_ids.values())

The 5 IGNORED event types (never appear in output CSVs, but affect normalization):
- camera_on (ID 5)   -> would be index 4/34
- half_start (ID 18) -> would be index 12/34  
- half_end (ID 34)   -> would be index 25/34
- starting_xi (ID 35) -> would be index 26/34
- tactical_shift (ID 36) -> would be index 27/34

These events are ignored AFTER normalization, so they are not present in the output CSVs.

This means there are 29 ACTUAL event types in the data, but the denominator is 34.
Decoding must use midpoint boundaries between consecutive VALID events.
"""

import pickle
import numpy as np
import argparse
from pathlib import Path

EVENT_TYPE_IDX = 0
SECOND_IDX = 9
MINUTE_IDX = 11
MAX_MINUTE = 140
MAX_SECOND = 60


# =============================================================================
# EVENT TYPE DEFINITIONS (from statsbomb_football_embeddings/tokenizer/config.py)
# =============================================================================

# ALL 34 event IDs as defined in the tokenizer (including ignored ones)
ALL_EVENT_IDS = {
    'ball_recovery': 2, 'dispossessed': 3, 'duel': 4, 'camera_on': 5, 'block': 6,
    'offside': 8, 'clearance': 9, 'interception': 10, 'dribble': 14, 'shot': 16,
    'pressure': 17, 'half_start': 18, 'substitution': 19, 'own_goal': 20, 'foul_won': 21,
    'foul_committed': 22, 'goalkeeper': 23, 'bad_behavior': 24, 'own_goal_for': 25,
    'player_on': 26, 'player_off': 27, 'shield': 28, 'pass': 30, '50_50': 33,
    'half_end': 34, 'starting_xi': 35, 'tactical_shift': 36, 'error': 37, 'miscontrol': 38,
    'dribbled_past': 39, 'injury_stoppage': 40, 'referee_ball_drop': 41, 'ball_receipt': 42, 'carry': 43
}

# The 5 ignored event types (from tokenizer config.py - ignore_event_type: True)
IGNORED_EVENT_IDS = {5, 18, 34, 35, 36}  # camera_on, half_start, half_end, starting_xi, tactical_shift

# The 29 actual event types that appear in embedded data
ACTUAL_EVENT_IDS = {name: id for name, id in ALL_EVENT_IDS.items() if id not in IGNORED_EVENT_IDS}

# Total number of categories used in normalization (ALL events, not just actual ones)
NUM_CATEGORIES = len(ALL_EVENT_IDS)  # 34


def build_event_mapping():
    """
    Build mapping from normalized values to event names.
    
    The tokenizer's CategoricalFeatureParser uses:
        categories = {value: index + 1 for index, value in enumerate(sorted(categories))}
        normalized = categories.get(val, 0) / num_categories
    
    So for event_id X at 0-indexed position P in sorted list:
        sorted_index = P + 1 (1-indexed)
        normalized_value = sorted_index / 34
    
    Returns:
        value_to_name: Dict mapping normalized_value -> event_name (only for actual events)
        sorted_events: List of (name, id) tuples sorted by id (only actual events)
        all_normalized: Dict mapping event_id -> normalized_value (all 34 events)
        boundaries: List of (lower_bound, upper_bound, event_name) for robust decoding
    """
    # Sort ALL event IDs by their numeric value (this is what the tokenizer does)
    all_sorted_by_id = sorted(ALL_EVENT_IDS.items(), key=lambda x: x[1])
    
    # Build mapping from event_id to normalized value (all 34)
    all_normalized = {}
    for position, (name, event_id) in enumerate(all_sorted_by_id):
        sorted_index = position + 1  # 1-indexed as per CategoricalFeatureParser
        normalized_value = sorted_index / NUM_CATEGORIES
        all_normalized[event_id] = normalized_value
    
    # Build mapping for actual events only (29 events)
    value_to_name = {}
    actual_sorted = []
    for name, event_id in all_sorted_by_id:
        if event_id not in IGNORED_EVENT_IDS:
            norm_val = all_normalized[event_id]
            value_to_name[norm_val] = name
            actual_sorted.append((name, event_id, norm_val))
    
    # Build decision boundaries using midpoints between consecutive ACTUAL events
    # This handles the gaps created by ignored events
    boundaries = []
    for i, (name, event_id, norm_val) in enumerate(actual_sorted):
        if i == 0:
            # First event: lower bound is 0 (or slightly above to exclude 0 = missing)
            lower = 0.001  # Values very close to 0 are treated as missing
        else:
            # Midpoint between this event and previous actual event
            prev_norm_val = actual_sorted[i - 1][2]
            lower = (prev_norm_val + norm_val) / 2
        
        if i == len(actual_sorted) - 1:
            # Last event: upper bound is infinity (handles values > 1)
            upper = float('inf')
        else:
            # Midpoint between this event and next actual event
            next_norm_val = actual_sorted[i + 1][2]
            upper = (norm_val + next_norm_val) / 2
        
        boundaries.append((lower, upper, name, norm_val))
    
    return value_to_name, actual_sorted, all_normalized, boundaries


def get_event_name(raw_value, boundaries, value_to_name=None):
    """
    Convert raw event type value to event name using boundary-based matching.
    
    This method uses the midpoint between consecutive valid events as decision
    boundaries, which correctly handles:
    - Gaps in normalized values due to ignored events
    - Slight numerical inaccuracies from diffusion model generation
    - Values slightly outside [0, 1] range
    
    Args:
        raw_value: The raw value from the decoded vector (index 0)
        boundaries: List of (lower_bound, upper_bound, event_name, exact_value) tuples
        value_to_name: Optional dict for reference (not used in matching)
    
    Returns:
        Tuple of (event_name, exact_normalized_value, distance_from_exact)
    """
    # Handle special cases
    if raw_value <= 0.001:
        # Value close to 0 indicates missing/invalid event type
        return ("MISSING", 0.0, raw_value)
    
    # Find the boundary interval that contains this value
    for lower, upper, name, exact_val in boundaries:
        if lower <= raw_value < upper:
            distance = abs(raw_value - exact_val)
            return (name, exact_val, distance)
    
    # Should not reach here if boundaries are correctly constructed
    return (f"UNKNOWN({raw_value:.4f})", raw_value, 0.0)


def find_shot_sequences(sequences, boundaries, all_normalized, prefer_ending=True):
    """
    Find sequences that contain shots, preferring those ending with shots.
    
    Args:
        sequences: Array of shape (n_sequences, seq_len, 128)
        boundaries: List of boundary tuples for decoding
        all_normalized: Dict mapping event_id -> normalized_value
        prefer_ending: If True, prioritize sequences ending with a shot
    
    Returns:
        List of (sequence_idx, shot_positions) tuples
    """
    # Get the exact normalized value for 'shot' event (ID 16)
    shot_norm_value = all_normalized[16]  # shot event ID is 16
    
    # Find the boundaries for shot event
    shot_lower = None
    shot_upper = None
    for lower, upper, name, exact_val in boundaries:
        if name == 'shot':
            shot_lower = lower
            shot_upper = upper
            break
    
    ending_with_shot = []
    containing_shot = []
    
    for seq_idx in range(len(sequences)):
        # Get event type values (index 0) for this sequence
        event_types = sequences[seq_idx, :, 0]
        
        # Find positions where shot occurs (using boundary matching)
        shot_positions = []
        for pos, val in enumerate(event_types):
            if shot_lower <= val < shot_upper:
                shot_positions.append(pos)
        
        if shot_positions:
            # Check if last event is a shot
            if shot_positions[-1] == len(event_types) - 1:
                ending_with_shot.append((seq_idx, shot_positions))
            else:
                containing_shot.append((seq_idx, shot_positions))
    
    print(f"\nFound {len(ending_with_shot)} sequences ending with shot")
    print(f"Found {len(containing_shot)} sequences containing shots (not ending)")
    
    # Prioritize sequences ending with shot
    if prefer_ending:
        result = ending_with_shot + containing_shot
    else:
        result = containing_shot + ending_with_shot
    
    return result


def extract_timestamp(event_vector):
    """
    Convert the normalized minute/second features into discrete timestamp values.

    Minute is denormalized to an integer in [0, 140].
    Second is denormalized to an integer in [0, 60].
    """
    minute = int(np.clip(np.round(event_vector[MINUTE_IDX] * MAX_MINUTE), 0, MAX_MINUTE))
    second = int(np.clip(np.round(event_vector[SECOND_IDX] * MAX_SECOND), 0, MAX_SECOND))
    total_seconds = minute * 60 + second

    # For display, fold second=60 into the next minute.
    display_minute = total_seconds // 60
    display_second = total_seconds % 60

    return {
        "minute": minute,
        "second": second,
        "display_minute": display_minute,
        "display_second": display_second,
        "total_seconds": total_seconds,
    }


def extract_timestamps(sequence):
    """Extract timestamps for all events in a sequence."""
    return [extract_timestamp(event) for event in sequence]


def count_time_violations(sequence):
    """
    Count non-monotonic timestamp violations in a sequence.

    A violation occurs when an event has an earlier (minute, second) timestamp
    than the event immediately before it.
    """
    timestamps = extract_timestamps(sequence)
    violation_count = 0
    violation_positions = []

    for i in range(1, len(timestamps)):
        if timestamps[i]["total_seconds"] < timestamps[i - 1]["total_seconds"]:
            violation_count += 1
            violation_positions.append(i)

    return violation_count, violation_positions, timestamps


def extract_event_sequence(sequence, boundaries, include_details=False):
    """
    Extract event names from a sequence using boundary-based decoding.
    
    Args:
        sequence: Array of shape (seq_len, 128)
        boundaries: List of boundary tuples for decoding
        include_details: If True, return detailed info including distances
    
    Returns:
        If include_details=False: List of event names
        If include_details=True: List of (event_name, exact_value, distance) tuples
    """
    event_types = sequence[:, EVENT_TYPE_IDX]
    results = [get_event_name(val, boundaries) for val in event_types]
    
    if include_details:
        return results
    else:
        return [name for name, _, _ in results]


def print_sequence(
    seq_idx,
    events,
    timestamps,
    shot_positions=None,
    title_prefix="",
    detailed_events=None,
    violation_count=0,
    violation_positions=None,
):
    """
    Pretty print a sequence of events.
    
    Args:
        seq_idx: Sequence index
        events: List of event names
        shot_positions: Optional list of positions where shots occur
        title_prefix: Prefix for the title
        detailed_events: Optional list of (name, exact_val, distance) for showing decoding quality
        violation_count: Number of non-monotonic time violations in the sequence
        violation_positions: Positions where time violations occur
    """
    print(f"\n{'='*80}")
    title = f"{title_prefix}Sequence {seq_idx}"
    if shot_positions:
        title += f" (shots at positions: {shot_positions})"
    print(title)
    print('='*80)
    print(f"Time violations: {violation_count}")
    if violation_positions:
        print(f"Violation positions: {violation_positions}")
    
    for i, event in enumerate(events):
        marker = " **SHOT**" if (shot_positions and i in shot_positions) else ""
        timestamp = timestamps[i]
        time_str = f"{timestamp['display_minute']:03d}:{timestamp['display_second']:02d}"
        if violation_positions and i in violation_positions:
            marker += " [TIME VIOLATION]"
        if detailed_events:
            name, exact_val, distance = detailed_events[i]
            quality = "✓" if distance < 0.01 else "~" if distance < 0.02 else "?"
            print(f"  {i+1:2d}. {event:20s} @ {time_str} {quality} (dist={distance:.4f}){marker}")
        else:
            print(f"  {i+1:2d}. {event:20s} @ {time_str}{marker}")


def print_decoding_reference(boundaries, all_normalized):
    """Print a reference table showing the exact decoding logic."""
    print("\n" + "=" * 90)
    print("EVENT TYPE DECODING REFERENCE")
    print("=" * 90)
    print(f"Total categories in tokenizer: {NUM_CATEGORIES} (including 5 ignored event types)")
    print(f"Actual event types in data: {len(ACTUAL_EVENT_IDS)}")
    print(f"\nIgnored events (affect normalization but never appear in data):")
    for name, id in ALL_EVENT_IDS.items():
        if id in IGNORED_EVENT_IDS:
            norm_val = all_normalized[id]
            print(f"  - {name} (ID {id}) -> normalized: {norm_val:.5f}")
    
    print(f"\nDecoding boundaries (value in [lower, upper) -> event):")
    print("-" * 90)
    print(f"{'Event Name':20s} {'ID':4s} {'Exact Value':12s} {'Lower Bound':12s} {'Upper Bound':12s}")
    print("-" * 90)
    for lower, upper, name, exact_val in boundaries:
        event_id = ACTUAL_EVENT_IDS.get(name, '?')
        upper_str = f"{upper:.5f}" if upper != float('inf') else "∞"
        print(f"{name:20s} {event_id:4d} {exact_val:.5f}      [{lower:.5f}, {upper_str})")
    print("-" * 90)


def analyze_decoding_quality(sequences, boundaries):
    """Analyze the quality of event type decoding across all sequences."""
    distances = []
    unknown_count = 0
    missing_count = 0
    
    for seq in sequences:
        event_types = seq[:, 0]
        for val in event_types:
            name, exact_val, distance = get_event_name(val, boundaries)
            if name.startswith("UNKNOWN"):
                unknown_count += 1
            elif name == "MISSING":
                missing_count += 1
            else:
                distances.append(distance)
    
    distances = np.array(distances)
    
    print("\n" + "=" * 60)
    print("DECODING QUALITY ANALYSIS")
    print("=" * 60)
    print(f"Total event values analyzed: {len(distances) + unknown_count + missing_count}")
    print(f"Successfully decoded: {len(distances)} ({len(distances)/(len(distances)+unknown_count+missing_count)*100:.1f}%)")
    print(f"Unknown (out of range): {unknown_count}")
    print(f"Missing (near zero): {missing_count}")
    
    if len(distances) > 0:
        print(f"\nDistance from exact normalized value:")
        print(f"  Mean:   {distances.mean():.5f}")
        print(f"  Std:    {distances.std():.5f}")
        print(f"  Median: {np.median(distances):.5f}")
        print(f"  Max:    {distances.max():.5f}")
        print(f"  <0.01:  {(distances < 0.01).sum()} ({(distances < 0.01).mean()*100:.1f}%)")
        print(f"  <0.02:  {(distances < 0.02).sum()} ({(distances < 0.02).mean()*100:.1f}%)")
        print(f"  <0.03:  {(distances < 0.03).sum()} ({(distances < 0.03).mean()*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract event sequences from decoded samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Decoding Logic:
  The tokenizer normalizes event types using: index / 34
  where index is the 1-based position in sorted(all 34 event IDs).
  
  5 event types are IGNORED (never in output but affect normalization):
    camera_on (5), half_start (18), half_end (34), starting_xi (35), tactical_shift (36)
  
  This script uses midpoint boundaries between consecutive VALID events
  for robust decoding of diffusion-generated samples with slight noise.
        """
    )
    parser.add_argument('--input', type=str, default='outputs/dit_decoded_samples.pkl',
                        help='Path to decoded samples pkl file')
    parser.add_argument('--num-random', type=int, default=10,
                        help='Number of random sequences to display')
    parser.add_argument('--num-shots', type=int, default=10,
                        help='Number of shot-related sequences to display (max)')
    parser.add_argument('--source', type=str, choices=['generated', 'real'], default='generated',
                        help='Use generated or real samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output file to save results')
    parser.add_argument('--show-details', action='store_true',
                        help='Show decoding distance for each event')
    parser.add_argument('--show-reference', action='store_true',
                        help='Show the full decoding reference table')
    parser.add_argument('--analyze-quality', action='store_true',
                        help='Analyze decoding quality across all sequences')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load decoded samples
    print(f"Loading samples from {args.input}...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    # Select source
    key = 'decoded_generated' if args.source == 'generated' else 'decoded_real'
    if key not in data:
        print(f"Error: '{key}' not found in pkl file. Available keys: {list(data.keys())}")
        return
    
    sequences = data[key]
    print(f"Loaded {len(sequences)} {args.source} sequences")
    print(f"Sequence shape: {sequences.shape}")  # Expected: (n_seq, 50, 128)
    
    # Build event mapping with correct normalization
    value_to_name, sorted_events, all_normalized, boundaries = build_event_mapping()
    
    # Print decoding reference if requested
    if args.show_reference:
        print_decoding_reference(boundaries, all_normalized)
    else:
        # Print compact mapping
        print(f"\nEvent Type Mapping (29 actual events, normalized using 34 categories):")
        print("-" * 70)
        for lower, upper, name, exact_val in boundaries:
            upper_str = f"{upper:.4f}" if upper != float('inf') else "∞"
            print(f"  [{lower:.4f}, {upper_str}) -> {name} (exact: {exact_val:.5f})")
    
    # Analyze decoding quality if requested
    if args.analyze_quality:
        analyze_decoding_quality(sequences, boundaries)
    
    output_lines = []
    total_time_violations = 0
    
    # === RANDOM SEQUENCES ===
    print(f"\n\n{'#'*80}")
    print(f"# RANDOM SEQUENCES ({args.num_random} sequences)")
    print(f"{'#'*80}")
    
    random_indices = np.random.choice(len(sequences), size=min(args.num_random, len(sequences)), replace=False)
    
    for seq_idx in random_indices:
        events = extract_event_sequence(sequences[seq_idx], boundaries, include_details=False)
        detailed = extract_event_sequence(sequences[seq_idx], boundaries, include_details=True) if args.show_details else None
        violation_count, violation_positions, timestamps = count_time_violations(sequences[seq_idx])
        total_time_violations += violation_count
        event_summary = " -> ".join(
            f"{event}@{ts['display_minute']:03d}:{ts['display_second']:02d}"
            for event, ts in zip(events, timestamps)
        )
        print_sequence(
            seq_idx,
            events,
            timestamps,
            title_prefix="[RANDOM] ",
            detailed_events=detailed,
            violation_count=violation_count,
            violation_positions=violation_positions,
        )
        output_lines.append(f"Random Sequence {seq_idx} [violations={violation_count}]: {event_summary}")
    
    # === SHOT-RELATED SEQUENCES ===
    print(f"\n\n{'#'*80}")
    print(f"# SHOT-RELATED SEQUENCES (up to {args.num_shots} sequences)")
    print(f"{'#'*80}")
    
    shot_sequences = find_shot_sequences(sequences, boundaries, all_normalized)
    
    if not shot_sequences:
        print("\nNo shot-related sequences found!")
    else:
        for i, (seq_idx, shot_positions) in enumerate(shot_sequences[:args.num_shots]):
            events = extract_event_sequence(sequences[seq_idx], boundaries, include_details=False)
            detailed = extract_event_sequence(sequences[seq_idx], boundaries, include_details=True) if args.show_details else None
            violation_count, violation_positions, timestamps = count_time_violations(sequences[seq_idx])
            total_time_violations += violation_count
            event_summary = " -> ".join(
                f"{event}@{ts['display_minute']:03d}:{ts['display_second']:02d}"
                for event, ts in zip(events, timestamps)
            )
            print_sequence(
                seq_idx,
                events,
                timestamps,
                shot_positions,
                title_prefix="[SHOT] ",
                detailed_events=detailed,
                violation_count=violation_count,
                violation_positions=violation_positions,
            )
            output_lines.append(
                f"Shot Sequence {seq_idx} (shots at {shot_positions}) [violations={violation_count}]: {event_summary}"
            )
    
    # === SUMMARY STATISTICS ===
    print(f"\n\n{'#'*80}")
    print(f"# EVENT TYPE DISTRIBUTION")
    print(f"{'#'*80}")
    
    # Count all event types across all sequences
    all_events = []
    all_distances = []
    for seq in sequences:
        detailed = extract_event_sequence(seq, boundaries, include_details=True)
        for name, exact_val, distance in detailed:
            all_events.append(name)
            all_distances.append(distance)
    
    # Count occurrences
    from collections import Counter
    event_counts = Counter(all_events)
    total_events = len(all_events)
    
    print(f"\nTotal events: {total_events}")
    overall_sequence_violations = [count_time_violations(seq)[0] for seq in sequences]
    print(f"Total time violations across all sequences: {sum(overall_sequence_violations)}")
    print(f"Sequences with at least one time violation: {sum(v > 0 for v in overall_sequence_violations)} / {len(sequences)}")
    print(f"\nEvent distribution:")
    for event, count in event_counts.most_common():
        pct = count / total_events * 100
        print(f"  {event:20s}: {count:6d} ({pct:5.2f}%)")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


def decode_single_value(raw_value):
    """
    Utility function to decode a single raw event type value.
    Useful for debugging or interactive use.
    
    Args:
        raw_value: The raw normalized value from index 0 of the event vector
    
    Returns:
        dict with 'event_name', 'event_id', 'exact_normalized', 'distance'
    """
    _, _, all_normalized, boundaries = build_event_mapping()
    name, exact_val, distance = get_event_name(raw_value, boundaries)
    
    # Find the event_id
    event_id = None
    for ename, eid in ACTUAL_EVENT_IDS.items():
        if ename == name:
            event_id = eid
            break
    
    return {
        'raw_value': raw_value,
        'event_name': name,
        'event_id': event_id,
        'exact_normalized': exact_val,
        'distance': distance,
    }


if __name__ == '__main__':
    main()
