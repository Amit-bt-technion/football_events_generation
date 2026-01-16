"""
Dataset module for loading and processing event sequences.
Uses the sampling functions from xG prediction repository.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class EventSequenceDataset(Dataset):
    """
    Dataset for event sequences ending with a shot (masked).
    Loads pre-computed embeddings and samples sequences.
    """
    
    def __init__(self, embeddings_path, sequence_length=50, split="train", 
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Args:
            embeddings_path: Path to pickled embeddings
            sequence_length: Length of sequences to generate
            split: One of 'train', 'val', 'test'
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            seed: Random seed for splitting
        """
        self.embeddings_path = embeddings_path
        self.sequence_length = sequence_length
        self.split = split
        
        # Load embeddings
        print(f"Loading embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # The embeddings pickle should contain:
        # - 'embeddings': dict mapping event_id to embedding vector
        # - 'sequences': list of sequences (list of event_ids ending with shot)
        # - 'metadata': additional information about events
        
        if isinstance(self.data, dict):
            self.embeddings = self.data.get('embeddings', {})
            self.sequences = self.data.get('sequences', [])
            self.metadata = self.data.get('metadata', {})
        else:
            # If it's just a list of sequences
            self.sequences = self.data
            self.embeddings = None
            self.metadata = {}
        
        # Filter sequences of appropriate length
        self.sequences = [
            seq for seq in self.sequences 
            if len(seq) == sequence_length
        ]
        
        print(f"Found {len(self.sequences)} sequences of length {sequence_length}")
        
        # Split data
        np.random.seed(seed)
        indices = np.random.permutation(len(self.sequences))
        
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        
        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        elif split == "test":
            self.indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"{split} split: {len(self.indices)} sequences")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns a sequence with the last event (shot) masked.
        
        Returns:
            sequence: Tensor of shape (sequence_length, embedding_dim)
                     Last event is masked (set to zeros)
            mask: Binary mask indicating which positions are valid (1) vs masked (0)
        """
        seq_idx = self.indices[idx]
        sequence = self.sequences[seq_idx]
        
        # Convert to tensor
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence).float()
        elif isinstance(sequence, list):
            sequence = torch.tensor(sequence).float()
        
        # Create mask: all 1s except last position (shot)
        mask = torch.ones(self.sequence_length)
        mask[-1] = 0
        
        # Mask the last event (set to zeros)
        masked_sequence = sequence.clone()
        masked_sequence[-1] = 0
        
        return {
            'sequence': masked_sequence,
            'mask': mask,
            'original': sequence  # Keep original for training target
        }


def create_dataloaders(args):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        args: Argument namespace with configuration
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = EventSequenceDataset(
        embeddings_path=args.embeddings_path,
        sequence_length=args.sequence_length,
        split="train",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        seed=args.seed
    )
    
    val_dataset = EventSequenceDataset(
        embeddings_path=args.embeddings_path,
        sequence_length=args.sequence_length,
        split="val",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        seed=args.seed
    )
    
    test_dataset = EventSequenceDataset(
        embeddings_path=args.embeddings_path,
        sequence_length=args.sequence_length,
        split="test",
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        seed=args.seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    return train_loader, val_loader, test_loader


def sample_sequences_from_xg_pipeline(cache_dir, sequence_length=50, num_samples=None):
    """
    Sample sequences using the xG prediction pipeline sampling functions.
    This is a placeholder - adapt based on actual xG prediction code structure.
    
    Args:
        cache_dir: Directory containing cached data
        sequence_length: Length of sequences to sample
        num_samples: Number of sequences to sample (None = all)
        
    Returns:
        List of sequences (numpy arrays of shape (sequence_length, embedding_dim))
    """
    # Import from xG prediction repository
    try:
        # Adjust import path based on actual xG repo structure
        from xg_prediction.dataset.sampling import sample_shot_sequences
        
        sequences = sample_shot_sequences(
            cache_dir=cache_dir,
            sequence_length=sequence_length,
            num_samples=num_samples
        )
        
        return sequences
        
    except ImportError:
        print("Warning: Could not import xG prediction sampling functions.")
        print("Using cached embeddings directly instead.")
        return None
