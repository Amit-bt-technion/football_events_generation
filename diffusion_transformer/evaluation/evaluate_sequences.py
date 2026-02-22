"""
Evaluation script for generated sequences using SequenceEvaluator.

This script processes CSV files from decoded sequence directories and evaluates them
for illegal transitions and non-monotonic timestamps.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from diffusion_transformer.evaluation.evaluator import SequenceEvaluator
from diffusion_transformer.utils import get_logger

logger = get_logger(__name__)


def load_sequence_from_csv(csv_path):
    """
    Load a sequence from a CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        numpy array of shape (seq_len, num_features)
    """
    df = pd.read_csv(csv_path)

    # Remove the 'event_index' column if present
    if 'event_index' in df.columns:
        df = df.drop(columns=['event_index'])

    # Convert to numpy array
    sequence = df.values

    return sequence


def evaluate_directory(sequences_dir, cache_dir, output_dir=None):
    """
    Evaluate all sequences in a directory.

    Args:
        sequences_dir: Directory containing sequence CSV files
        cache_dir: Cache directory for transition matrix
        output_dir: Directory to save results (defaults to sequences_dir)

    Returns:
        Dictionary with evaluation results
    """
    if output_dir is None:
        output_dir = sequences_dir

    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator
    logger.info(f"Initializing SequenceEvaluator with cache_dir: {cache_dir}")
    evaluator = SequenceEvaluator(
        cache_dir=cache_dir,
        transition_tolerance=0.0,  # Strict: no illegal transitions allowed
        time_tolerance=0  # Strict: monotonic timestamps required
    )

    # Find all sequence CSV files
    sequences_dir_path = Path(sequences_dir)
    sequence_files = sorted(sequences_dir_path.glob('sequence_*.csv'))

    if not sequence_files:
        logger.warning(f"No sequence files found in {sequences_dir}")
        return None

    logger.info(f"Found {len(sequence_files)} sequence files in {sequences_dir}")

    # Load all sequences
    sequences = []
    for seq_file in tqdm(sequence_files, desc="Loading sequences"):
        try:
            sequence = load_sequence_from_csv(seq_file)
            sequences.append(sequence)
        except Exception as e:
            logger.error(f"Error loading {seq_file}: {e}")

    if not sequences:
        logger.error("No sequences loaded successfully")
        return None

    # Stack into batch
    sequences_batch = np.array(sequences)
    logger.info(f"Loaded sequences shape: {sequences_batch.shape}")

    # Evaluate batch
    logger.info("Evaluating sequences...")
    results = evaluator.evaluate_batch(sequences_batch)

    # Add metadata
    results['sequences_dir'] = str(sequences_dir)
    results['num_files'] = len(sequence_files)
    results['sequence_shape'] = sequences_batch.shape

    # Save results as JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    # Convert all values recursively
    def make_serializable(data):
        if isinstance(data, dict):
            return {k: make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [make_serializable(item) for item in data]
        else:
            return convert_to_serializable(data)

    serializable_results = make_serializable(results)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Log summary
    logger.info("="*60)
    logger.info(f"EVALUATION SUMMARY for {sequences_dir_path.name}")
    logger.info("="*60)
    logger.info(f"Total sequences: {results['total_sequences']}")
    logger.info(f"Passed: {results['passed_sequences']} ({results['pass_rate']:.2%})")
    logger.info(f"Failed: {results['failed_sequences']}")
    logger.info(f"Total transition violations: {results['total_transition_violations']}")
    logger.info(f"Total time violations: {results['total_time_violations']}")
    logger.info("="*60)

    return results


def main():
    """Main evaluation function."""
    # Define paths
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / 'outputs' / 'evaluation'
    cache_dir = project_root / 'cache'

    logger.info("="*60)
    logger.info("SEQUENCE EVALUATION SCRIPT")
    logger.info("="*60)
    logger.info(f"Outputs directory: {outputs_dir}")
    logger.info(f"Cache directory: {cache_dir}")

    # Find all decoded_sequences_* directories
    decoded_dirs = sorted([d for d in outputs_dir.iterdir()
                          if d.is_dir() and d.name.startswith('decoded_sequences_')])

    if not decoded_dirs:
        logger.error(f"No decoded_sequences_* directories found in {outputs_dir}")
        return

    logger.info(f"Found {len(decoded_dirs)} directories to evaluate:")
    for d in decoded_dirs:
        logger.info(f"  - {d.name}")
    logger.info("")

    # Evaluate each directory
    all_results = {}
    for decoded_dir in decoded_dirs:
        logger.info(f"\nProcessing: {decoded_dir.name}")
        logger.info("-"*60)

        try:
            results = evaluate_directory(
                sequences_dir=decoded_dir,
                cache_dir=cache_dir,
                output_dir=decoded_dir  # Save results in the same directory
            )

            if results:
                all_results[decoded_dir.name] = {
                    'total_sequences': results['total_sequences'],
                    'passed_sequences': results['passed_sequences'],
                    'failed_sequences': results['failed_sequences'],
                    'pass_rate': results['pass_rate'],
                    'total_transition_violations': results['total_transition_violations'],
                    'total_time_violations': results['total_time_violations']
                }
        except Exception as e:
            logger.error(f"Error evaluating {decoded_dir.name}: {e}", exc_info=True)

    # Save combined summary
    if all_results:
        summary_path = outputs_dir / 'evaluation_summary_all.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("ALL EVALUATIONS COMPLETED")
        logger.info("="*60)
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("\nOverall Summary:")
        for dir_name, summary in all_results.items():
            logger.info(f"\n{dir_name}:")
            logger.info(f"  Pass rate: {summary['pass_rate']:.2%}")
            logger.info(f"  Passed/Total: {summary['passed_sequences']}/{summary['total_sequences']}")
            logger.info(f"  Transition violations: {summary['total_transition_violations']}")
            logger.info(f"  Time violations: {summary['total_time_violations']}")
    else:
        logger.warning("No results collected")


if __name__ == '__main__':
    main()

