"""
Unit tests for diffusion transformer pipeline.
Run with: pytest tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import pickle

from diffusion_transformer.models.dit import DiffusionTransformer
from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.data.dataset import EventSequenceDataset


class TestDiffusionTransformer:
    """Test the DiT model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = DiffusionTransformer(
            input_dim=32,
            model_dim=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=50
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass works."""
        model = DiffusionTransformer(
            input_dim=32,
            model_dim=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=50
        )
        
        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 32)
        t = torch.randint(0, 1000, (batch_size,))
        
        output = model(x, t)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = DiffusionTransformer(
            input_dim=32,
            model_dim=128,
            num_layers=2,
            num_heads=4
        )
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params > 0
        print(f"Model parameters: {num_params:,}")


class TestDiffusionProcess:
    """Test the diffusion process."""
    
    def test_noise_schedules(self):
        """Test different noise schedules."""
        for schedule in ["linear", "cosine", "quadratic"]:
            diffusion = DiffusionProcess(
                num_timesteps=100,
                schedule_type=schedule,
                device="cpu"
            )
            
            assert diffusion.betas.shape == (100,)
            assert torch.all(diffusion.betas > 0)
            assert torch.all(diffusion.betas < 1)
    
    def test_forward_diffusion(self):
        """Test forward diffusion process."""
        diffusion = DiffusionProcess(
            num_timesteps=100,
            schedule_type="linear",
            device="cpu"
        )
        
        x_0 = torch.randn(4, 50, 32)
        t = torch.randint(0, 100, (4,))
        
        x_t = diffusion.q_sample(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert not torch.isnan(x_t).any()
    
    def test_sampling(self):
        """Test sampling process."""
        model = DiffusionTransformer(
            input_dim=32,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=50
        )
        model.eval()
        
        diffusion = DiffusionProcess(
            num_timesteps=10,  # Small for testing
            schedule_type="linear",
            device="cpu"
        )
        
        shape = (2, 50, 32)
        samples = diffusion.p_sample_loop(model, shape, progress=False)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
    
    def test_ddim_sampling(self):
        """Test DDIM sampling."""
        model = DiffusionTransformer(
            input_dim=32,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=50
        )
        model.eval()
        
        diffusion = DiffusionProcess(
            num_timesteps=20,
            schedule_type="linear",
            device="cpu"
        )
        
        shape = (2, 50, 32)
        samples = diffusion.ddim_sample(
            model, shape, ddim_steps=5, progress=False
        )
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()


class TestDataset:
    """Test the dataset."""
    
    def create_dummy_data(self, tmp_path):
        """Create dummy embeddings for testing."""
        # Create dummy sequences
        num_sequences = 100
        sequences = []
        
        for _ in range(num_sequences):
            seq = np.random.randn(50, 32).astype(np.float32)
            sequences.append(seq)
        
        data = {
            'sequences': sequences,
            'metadata': {}
        }
        
        # Save to temp file
        pkl_path = tmp_path / "embeddings.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        return str(pkl_path)
    
    def test_dataset_creation(self, tmp_path):
        """Test dataset can be created."""
        pkl_path = self.create_dummy_data(tmp_path)
        
        dataset = EventSequenceDataset(
            embeddings_path=pkl_path,
            sequence_length=50,
            split="train",
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )
        
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__."""
        pkl_path = self.create_dummy_data(tmp_path)
        
        dataset = EventSequenceDataset(
            embeddings_path=pkl_path,
            sequence_length=50,
            split="train"
        )
        
        item = dataset[0]
        
        assert 'sequence' in item
        assert 'mask' in item
        assert 'original' in item
        
        assert item['sequence'].shape == (50, 32)
        assert item['mask'].shape == (50,)
        assert item['original'].shape == (50, 32)
        
        # Check last position is masked
        assert item['mask'][-1] == 0
        assert torch.all(item['sequence'][-1] == 0)
    
    def test_dataset_splits(self, tmp_path):
        """Test train/val/test splits."""
        pkl_path = self.create_dummy_data(tmp_path)
        
        train_ds = EventSequenceDataset(pkl_path, split="train")
        val_ds = EventSequenceDataset(pkl_path, split="val")
        test_ds = EventSequenceDataset(pkl_path, split="test")
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert total == 100  # We created 100 sequences


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self, tmp_path):
        """Test one training step."""
        # Create dummy data
        pkl_path = self.create_dummy_data_file(tmp_path)
        
        # Create model and diffusion
        model = DiffusionTransformer(
            input_dim=32,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=50
        )
        
        diffusion = DiffusionProcess(
            num_timesteps=100,
            schedule_type="linear",
            device="cpu"
        )
        
        # Create dataset and loader
        dataset = EventSequenceDataset(pkl_path, split="train")
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Get batch
        batch = next(iter(loader))
        x_0 = batch['original']
        mask = batch['mask']
        
        # Training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        t = torch.randint(0, 100, (len(x_0),))
        noise = torch.randn_like(x_0)
        x_t = diffusion.q_sample(x_0, t, noise)
        
        noise_pred = model(x_t, t, mask=(1 - mask).bool())
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def create_dummy_data_file(self, tmp_path):
        """Helper to create dummy data."""
        sequences = [np.random.randn(50, 32).astype(np.float32) for _ in range(20)]
        data = {'sequences': sequences}
        
        pkl_path = tmp_path / "embeddings.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        return str(pkl_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
