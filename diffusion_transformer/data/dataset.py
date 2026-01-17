"""
dataset.py
---------
Custom dataset for loading and preparing sequences of football events with flexible sampling strategies.
"""

from __future__ import annotations
import argparse
import logging
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm


# Task registry to store sampling and labeling functions
TASK_REGISTRY = {}


def register_task_logic(task_name):
    """Decorator to register task functions in the global registry."""

    def decorator(func):
        TASK_REGISTRY[task_name] = {}
        TASK_REGISTRY[task_name]["logic"] = func
        return func

    return decorator


def register_task_item_getter(task_name):
    """Decorator to register task getitem dunder in the global registry."""

    def decorator(func):
        TASK_REGISTRY[task_name]["getitem"] = func
        return func

    return decorator


class EventSequenceDataset(Dataset):
    """
    Dataset for creating samples of event sequences from embedded football events.
    Supports different sampling strategies based on the task.
    """

    def __init__(self, args, events_dict: dict[str, np.ndarray], embeddings_dict: dict[str, np.ndarray], match_ids=None, shuffle=False):
        """
        Initialize the EventSequenceDataset.

        :param args: Arguments passed to the training script
        :param events_dict: Dictionary of event DataFrames keyed by match_id
        :param embeddings_dict: dictionary of precomputed embeddings keyed by match_id
        :param match_ids: Optional list of match IDs to filter by
        :param shuffle: Whether to shuffle the samples
        """
        self.logger = logging.getLogger(__name__)
        self.events_dict = events_dict
        self.sequence_length = args.sequence_length
        self.min_gap = args.min_gap
        self.max_gap = args.max_gap
        self.task = args.task
        self.task_params = args.task_params or {}

        # Check if task exists in registry
        assert args.task in TASK_REGISTRY, f"Task '{args.task}' not found in registry. Available tasks: {list(TASK_REGISTRY.keys())}"

        # Get unique match IDs
        self.match_ids = match_ids
        self.logger.info(f"Processing {len(self.match_ids)} unique matches for task '{args.task}'")

        # Group events by match_id
        self.num_match_events = {}
        self.embeddings = {}

        # Organize data by match
        for match_id in tqdm(self.match_ids, desc="Organizing match data", disable=not args.verbose):

            # If no precomputed embeddings are provided, raise ValueError
            if match_id not in embeddings_dict:
                raise ValueError(f"No embeddings for match {match_id}.")

            # Set events and embeddings properties
            self.embeddings[match_id] = embeddings_dict[match_id]
            self.num_match_events[match_id] = len(self.embeddings[match_id])

        # Generate samples using the task-specific function
        self.samples = self._generate_samples(
            max_samples_per_match=args.max_samples_per_match,
            max_total_samples=args.max_samples_total,
            verbose=args.verbose
        )

        if shuffle:
            random.shuffle(self.samples)

        self.logger.info(f"Created dataset with {len(self.samples)} samples for task '{args.task}'")

    def _generate_samples(
            self,
            max_samples_per_match: int | None = None,
            max_total_samples: int | None = None,
            verbose: bool = True
    ) -> list[tuple]:
        """
        Generate samples using the task-specific sampling function.

        Returns:
            list of samples as defined by the task-specific function
        """
        # Get the task-specific sampling function
        sampling_func = TASK_REGISTRY[self.task]["logic"]

        samples = []
        total_samples = 0

        for match_id in tqdm(self.match_ids, desc=f"Generating samples for {self.task}", disable=not verbose):
            num_events = self.num_match_events[match_id]

            # Skip matches that don't have enough events for the minimum sequence length
            if num_events < self.sequence_length:
                self.logger.debug(f"Skipping match {match_id}: not enough events")
                continue



            # Call the task-specific sampling function
            match_samples = sampling_func(
                match_id=match_id,
                num_events=num_events,
                sequence_length=self.sequence_length,
                min_gap=self.min_gap,
                max_gap=self.max_gap,
                max_samples=max_samples_per_match,
                **self.task_params
            )

            samples.extend(match_samples)
            total_samples += len(match_samples)

            if max_total_samples is not None and total_samples >= max_total_samples:
                self.logger.info(f"Reached maximum total samples ({max_total_samples})")
                samples = samples[:max_total_samples]
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            A tuple containing input tensor(s) and target tensor(s) as defined by the task
        """
        sample = self.samples[idx]
        match_id = sample[0]  # First element is always match_id
        return TASK_REGISTRY[self.task]["getitem"](sample, match_id, self.embeddings, self.sequence_length,
                                                   events_dict=self.events_dict
                                                   )


# Register task functions


@register_task_logic("full")
def sample_all_sequences(
        match_id: str,
        num_events: int,
        sequence_length: int,
        max_samples: int | None = None,
        **kwargs
) -> list[tuple]:
    """
    Generate samples for dominating team prediction.
    Used for both classification and regression tasks.

    Args:
        match_id: Match ID
        num_events: Number of events in the match
        sequence_length: Number of events in each sequence
        max_samples: Maximum number of samples to generate

    Returns:
        list of tuples: (match_id, seq_start)
    """
    samples = []

    # Skip matches that don't have enough events
    if num_events < sequence_length:
        return samples

    # Maximum starting position to ensure we have enough events for the sequence and the target
    max_start = num_events - sequence_length

    # Generate random starting positions
    possible_starts = list(range(max_start + 1))
    random.shuffle(possible_starts)

    # Limit samples per match if specified
    if max_samples is not None:
        possible_starts = possible_starts[:max_samples]

    for seq_start in possible_starts:
        samples.append((match_id, seq_start))

    return samples


@register_task_item_getter("full")
def get_item_for_dominating_team_classification(
        sample: tuple,
        match_id: str,
        embeddings_dict: dict[str, np.ndarray],
        sequence_length: int,
        events_dict: dict[str, np.ndarray],
) -> Tensor:
    """
    Getitem logic for next event prediction.

    Args:
        sample: The sample matching the requested idx
        match_id: Match ID
        embeddings_dict: Match events embeddings
        sequence_length: Number of events in each sequence
        events_dict: A dictionary of event keys and event DataFrame values

    Returns:
        tuple: (sample, label)
    """
    seq_start = sample[1]

    # Get embeddings for sequence and next event
    seq = embeddings_dict[match_id][seq_start:seq_start + sequence_length]

    # Convert to tensors
    X = torch.tensor(seq, dtype=torch.float32)

    return X



def is_shot_event(event_vector: np.ndarray) -> bool:
    """
    Check if an event is a shot event (type_id = 16).

    Args:
        event_vector: Tokenized event vector

    Returns:
        True if the event is a shot, False otherwise
    """
    # The event type is stored at index 0 and shot events have value 16
    # From tokenized_eventpy.py: event_ids['shot'] = 16
    # The list of event_ids values is used in CategoricalFeatureParser
    # Shot event_id 16 is the 10th unique value in the sorted list, so normalized as (position)/total
    # Let's use a more robust approach - check if it's close to the expected normalized value
    shot_event_normalized = 13 / 44  # Shot is the 10th value out of 44 possible event types
    return abs(event_vector[0] - shot_event_normalized) < 0.02  # Allow some tolerance


def extract_goal_label_from_shot(event_vector: np.ndarray) -> int:
    """
    Extract binary goal label from shot event outcome.id field.

    Args:
        event_vector: Tokenized shot event vector

    Returns:
        1 if goal, 0 if not goal
    """
    # shot.outcome.id is at index 83 in the tokenized vector
    # The outcome values are [96, 97, 98, 99, 100, 101, 115, 116]
    # Value 97 represents a goal and is the 2nd value (index 1), so it's parsed as 2/8 = 0.25
    outcome_value = event_vector[83]

    # Check if it's close to 0.25 (goal value)
    goal_value_normalized = 2 / 8  # 0.25
    return 1 if abs(outcome_value - goal_value_normalized) < 0.01 else 0


@register_task_logic("xg_prediction")
def sample_xg_sequences(
        match_id: str,
        num_events: int,
        sequence_length: int,
        min_gap: int,
        max_gap: int | None = None,
        max_samples: int | None = None,
        **kwargs
) -> list[tuple]:
    """
    Generate sequences ending with shot events for xG prediction and MLP baseline.

    Args:
        match_id: Match ID
        num_events: Number of events in the match
        sequence_length: Number of events in each sequence (200 for xG, 1 for MLP baseline)
        min_gap: Minimum number of events between sequences (not used for xG/MLP)
        max_gap: Maximum number of events between sequences (not used for xG/MLP)
        max_samples: Maximum number of samples to generate

    Returns:
        list of tuples: (match_id, sequence_end_index, goal_label)
    """
    samples = []

    # Skip matches that don't have enough events
    if num_events < sequence_length:
        return samples

    # Find all shot events in the match
    shot_indices = []
    events_dict = kwargs.get('events_dict', {})

    if match_id in events_dict:
        match_events = events_dict[match_id]

        # For MLP baseline (sequence_length=1), start from 0; for xG (sequence_length=200), start from sequence_length-1
        start_idx = max(0, sequence_length - 1)
        for i in range(start_idx, num_events):
            event_vector = match_events[i]
            if is_shot_event(event_vector):
                goal_label = extract_goal_label_from_shot(event_vector)
                shot_indices.append((i, goal_label))

    # Limit samples if specified
    if max_samples is not None and len(shot_indices) > max_samples:
        # Take a random sample to maintain balance
        random.shuffle(shot_indices)
        shot_indices = shot_indices[:max_samples]

    # Create samples: sequences ending with shot events
    for shot_idx, goal_label in shot_indices:
        samples.append((match_id, shot_idx, goal_label))

    return samples


@register_task_item_getter("xg_prediction")
def get_item_for_xg_prediction(
        sample: tuple,
        match_id: str,
        embeddings_dict: dict[str, np.ndarray],
        sequence_length: int,
        events_dict: dict[str, np.ndarray],
        **kwargs
) -> tuple:
    """
    Getitem logic for xG prediction and MLP baseline.

    Args:
        sample: The sample tuple (match_id, sequence_end_index, goal_label)
        match_id: Match ID
        embeddings_dict: Match events embeddings (already masked)
        sequence_length: Number of events in each sequence (200 for xG, 1 for MLP baseline)
        events_dict: dictionary of event arrays by match_id

    Returns:
        tuple: (sequence_tensor, goal_label_tensor)
    """
    sequence_end_idx, goal_label = sample[1], sample[2]

    # Extract sequence ending with the shot event
    # For MLP baseline (sequence_length=1): sequence_start = sequence_end_idx - 1 + 1 = sequence_end_idx
    # This gives us seq = embeddings_dict[match_id][sequence_end_idx:sequence_end_idx + 1] (single shot event)
    # For xG (sequence_length=200): sequence_start = sequence_end_idx - 199, giving full sequence
    sequence_start = sequence_end_idx - sequence_length + 1
    seq = embeddings_dict[match_id][sequence_start:sequence_end_idx + 1]

    # Convert to tensors
    X = torch.tensor(seq, dtype=torch.float32)
    y = torch.tensor(goal_label, dtype=torch.float32)  # Float for regression output

    return X, y


def create_dataloaders(args, events_dict: dict, embeddings_dict: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    :param args: The arguments for the pipeline.
    :param events_dict: Dictionary containing event data.
    :param embeddings_dict: Dictionary containing event embeddings.

    Args should contain:
        task: Name of the task to determine sampling strategy
        sequence_length: Number of events in each sequence
        min_gap: Minimum number of events between sequences
        max_gap: Maximum number of events between sequences
        train_split: Proportion of matches to use for training
        val_split: Proportion of matches to use for validation
        batch_size: Batch size for data loaders
        max_samples_per_match: Maximum samples to generate per match
        max_samples_total: Maximum total samples
        num_workers: Number of workers for data loading
        task_params: Additional parameters specific to the task
        verbose: Whether to show progress bars

    Returns:
        tuple of (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)

    # Merge task parameters with events_df for tasks that need it
    if args.task_params is None:
        args.task_params = {}

    # For tasks that need the events_df
    if args.task == "event_type_classification":
        args.task_params['events_df'] = events_dict

    # For xG prediction task, pass events_dict for shot detection
    if args.task in ["xg_prediction", "mlp_baseline"]:
        args.task_params['events_dict'] = events_dict

    # Get unique match IDs and shuffle them
    match_ids = list(events_dict.keys())
    random.shuffle(match_ids)

    # Split match IDs into train, validation, and test sets
    num_matches = len(match_ids)
    train_size = int(args.train_split * num_matches)
    val_size = int(args.val_split * num_matches)

    train_match_ids = match_ids[:train_size]
    val_match_ids = match_ids[train_size:train_size + val_size]
    test_match_ids = match_ids[train_size + val_size:]

    logger.info(f"Split {num_matches} matches into {len(train_match_ids)} train, "
                f"{len(val_match_ids)} validation, and {len(test_match_ids)} test")

    # Create datasets
    train_dataset = EventSequenceDataset(
        args,
        events_dict=events_dict,
        embeddings_dict=embeddings_dict,
        match_ids=train_match_ids,
        shuffle=True
    )

    # Use fewer samples for validation
    val_args = argparse.Namespace(**vars(args))
    val_args.max_samples_per_match = args.max_samples_per_match // 2
    val_args.max_samples_total = args.max_samples_total // 10

    val_dataset = EventSequenceDataset(
        val_args,
        events_dict=events_dict,
        embeddings_dict=embeddings_dict,
        match_ids=val_match_ids,
        shuffle=True
    )

    # Use fewer samples for testing
    test_args = argparse.Namespace(**vars(args))
    test_args.max_samples_per_match = args.max_samples_per_match // 2
    test_args.max_samples_total = args.max_samples_total // 10

    test_dataset = EventSequenceDataset(
        test_args,
        events_dict=events_dict,
        embeddings_dict=embeddings_dict,
        match_ids=test_match_ids,
        shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader






