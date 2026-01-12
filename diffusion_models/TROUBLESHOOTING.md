# Troubleshooting Guide

Common issues and their solutions when running the Diffusion Transformer pipeline.

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'diffusion_transformer'"

**Cause**: Package not installed or wrong directory

**Solutions**:
```bash
# Option 1: Install as package
pip install -e .

# Option 2: Run from root directory
cd /path/to/diffusion_transformer
python -m diffusion_transformer.main --task train

# Option 3: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/diffusion_transformer"
```

### Issue: "ImportError: cannot import name 'DiffusionTransformer'"

**Cause**: Missing `__init__.py` files

**Solution**: Ensure all subdirectories have `__init__.py`:
```bash
touch diffusion_transformer/__init__.py
touch diffusion_transformer/data/__init__.py
touch diffusion_transformer/models/__init__.py
touch diffusion_transformer/training/__init__.py
touch diffusion_transformer/evaluation/__init__.py
touch diffusion_transformer/visualization/__init__.py
```

## Data Issues

### Issue: "FileNotFoundError: embeddings.pkl not found"

**Cause**: Embeddings file missing or wrong path

**Solutions**:
```bash
# Check file exists
ls -lh cache/embeddings/embeddings.pkl

# Create directory if needed
mkdir -p cache/embeddings

# Specify custom path
python -m diffusion_transformer.main \
    --task train \
    --embeddings_path /path/to/your/embeddings.pkl
```

### Issue: "ValueError: Invalid pickle format"

**Cause**: Corrupted pickle file or wrong format

**Solution**: Verify pickle structure:
```python
import pickle
with open('cache/embeddings/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    print(type(data))
    print(data.keys() if isinstance(data, dict) else 'List of length', len(data))
```

Expected structure:
```python
{
    'sequences': [array(50, 32), array(50, 32), ...],  # List of sequences
    'metadata': {...}  # Optional
}
```

### Issue: "RuntimeError: Found 0 sequences of length 50"

**Cause**: No sequences match the requested length

**Solutions**:
```bash
# Check available sequence lengths
python -c "
import pickle
with open('cache/embeddings/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    seqs = data.get('sequences', data)
    lengths = [len(s) for s in seqs]
    print(f'Available lengths: {set(lengths)}')
    print(f'Counts: {[(l, lengths.count(l)) for l in set(lengths)]}')
"

# Adjust sequence_length parameter
python -m diffusion_transformer.main \
    --task train \
    --sequence_length 30  # Use actual length
```

## Training Issues

### Issue: "CUDA out of memory"

**Cause**: Model too large for GPU

**Solutions**:
```bash
# Reduce batch size
--batch_size 32

# Reduce model size
--model_dim 256 --num_layers 4

# Use gradient accumulation (modify trainer.py)
# Or use CPU (slow)
--device cpu
```

### Issue: "Loss is NaN"

**Cause**: Numerical instability

**Solutions**:
```bash
# Reduce learning rate
--learning_rate 1e-5

# Increase gradient clipping
--grad_clip 0.5

# Check for NaN in data
python -c "
import pickle, numpy as np
with open('cache/embeddings/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    seqs = data.get('sequences', data)
    for i, s in enumerate(seqs):
        if np.isnan(s).any():
            print(f'NaN in sequence {i}')
"

# Use different noise schedule
--noise_schedule cosine  # More stable than linear
```

### Issue: "Training is very slow"

**Causes & Solutions**:

1. **CPU instead of GPU**:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# If False, install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

2. **Too many workers**:
```bash
--num_workers 0  # Try single-threaded
```

3. **Large model**:
```bash
--model_dim 256 --num_layers 4  # Start small
```

4. **Too many timesteps for eval**:
```bash
--ddim_steps 20  # Reduce sampling steps
--eval_every 10  # Evaluate less frequently
```

### Issue: "Validation loss not decreasing"

**Causes & Solutions**:

1. **Overfitting**:
```bash
--dropout 0.2  # Increase dropout
--weight_decay 0.05  # Increase regularization
```

2. **Underfitting**:
```bash
--model_dim 768  # Larger model
--num_layers 12
--num_epochs 200  # Train longer
```

3. **Bad hyperparameters**:
```bash
--learning_rate 5e-5  # Try different LR
--noise_schedule cosine  # Try different schedule
```

## Generation Issues

### Issue: "Generated samples look random"

**Cause**: Model not trained or undertrained

**Solutions**:
- Train longer (100+ epochs)
- Check training loss is decreasing
- Verify checkpoint loaded correctly
- Increase model capacity

### Issue: "All generated samples look the same" (Mode Collapse)

**Cause**: Poor diversity in generation

**Solutions**:
```bash
# Train with diversity-focused loss (future work)
# Increase temperature (modify diffusion.py)
# Use different noise schedule
--noise_schedule cosine

# Check diversity metrics
--diversity_weight 0.8  # Prioritize diversity
```

### Issue: "Generation takes too long"

**Solutions**:
```bash
# Use DDIM
--ddim_steps 25  # Reduce steps (min ~20)

# Generate fewer samples
--num_gen_samples 50

# Use smaller model for testing
--model_dim 256 --num_layers 4
```

## Evaluation Issues

### Issue: "Frechet Distance is very high (>100)"

**Interpretations**:
- Normal for early training
- May indicate distribution mismatch
- Could be due to small sample size

**Solutions**:
```bash
# Evaluate with more samples
--num_eval_samples 2000

# Train longer
--num_epochs 200

# Check if real data is normalized
```

### Issue: "xG metrics not working"

**Cause**: xG model not loaded (placeholder code)

**Solution**: Implement xG model loading in `evaluator.py`:
```python
def load_xg_model(self, model_path):
    # Load your actual xG model
    from your_xg_module import XGModel
    model = XGModel(...)
    model.load_state_dict(torch.load(model_path))
    return model
```

## Visualization Issues

### Issue: "No plots generated"

**Cause**: Missing data or errors in plotting

**Solutions**:
```bash
# Check output directory
ls -lh outputs/

# Run visualize task explicitly
python -m diffusion_transformer.main --task visualize

# Check for errors in logs
```

### Issue: "Plots look wrong"

**Cause**: Data format mismatch

**Solution**: Verify sample shapes:
```python
import pickle
with open('outputs/generated_samples.pkl', 'rb') as f:
    data = pickle.load(f)
    samples = data.get('samples', data.get('generated'))
    print(f'Shape: {samples.shape}')  # Should be (N, 50, 32)
```

### Issue: "Cannot create animation"

**Cause**: Missing Pillow or trajectory data

**Solutions**:
```bash
# Install Pillow
pip install Pillow

# Check trajectory exists
python -c "
import pickle
with open('outputs/generated_samples.pkl', 'rb') as f:
    data = pickle.load(f)
    print('trajectory' in data)
"
```

## Memory Issues

### Issue: "System runs out of RAM"

**Solutions**:
```bash
# Reduce number of cached samples
--num_cached_samples 5

# Reduce evaluation sample size
--num_eval_samples 500

# Use fewer data loading workers
--num_workers 0

# Clear cache periodically (modify code)
```

### Issue: "Checkpoint files too large"

**Solutions**:
```bash
# Save less frequently
--save_every 20

# Delete old checkpoints manually
rm models/checkpoint_epoch_*.pt

# Keep only best model
```

## Checkpoint Issues

### Issue: "Cannot load checkpoint"

**Cause**: Version mismatch or corruption

**Solutions**:
```bash
# Check checkpoint integrity
python -c "
import torch
ckpt = torch.load('models/best_model.pt', map_location='cpu')
print(ckpt.keys())
"

# Try loading with weights_only
# Modify trainer.py load_checkpoint to use weights_only=False
```

### Issue: "Resume training fails"

**Cause**: State dict mismatch

**Solution**: Check model architecture matches:
```python
# In checkpoint
ckpt = torch.load('models/best_model.pt')
print(ckpt['args'])  # Check saved args

# Run with same args
python -m diffusion_transformer.main \
    --task train \
    --checkpoint models/best_model.pt \
    --model_dim 512 \  # Match saved args
    --num_layers 8
```

## Performance Issues

### Issue: "Training slower than expected"

**Checklist**:
- [ ] Using GPU? (`--device cuda`)
- [ ] Mixed precision enabled? (automatic if CUDA)
- [ ] Reasonable batch size? (64-128)
- [ ] Not too many workers? (4-8)
- [ ] No CPU bottleneck? (check with `nvidia-smi`)

### Issue: "Evaluation takes forever"

**Solutions**:
```bash
# Use DDIM instead of DDPM
--ddim_steps 25  # Much faster

# Evaluate fewer samples
--num_eval_samples 500

# Evaluate less frequently
--eval_every 10
```

## Integration Issues

### Issue: "Cannot load autoencoder"

**Cause**: Placeholder code not implemented

**Solution**: Add autoencoder loading in `generator.py`:
```python
def load_autoencoder(self, model_path):
    from embed_ball.models import TransformerAutoEncoder
    model = TransformerAutoEncoder(...)
    model.load_state_dict(torch.load(model_path))
    return model.eval()
```

### Issue: "xG prediction integration not working"

**Cause**: Import path issues

**Solution**: Add xG repo to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/xG_prediction_model"
```

## Getting More Help

1. **Check logs**: Training prints detailed progress
2. **Run tests**: `pytest tests/test_pipeline.py -v`
3. **Verify data**: Check pickle file structure
4. **Start small**: Test with tiny model first
5. **Check GPU**: `nvidia-smi` to monitor usage

## Common Error Messages

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| "KeyError: 'sequences'" | Wrong data format | Check pickle structure |
| "RuntimeError: CUDA error" | GPU memory | Reduce batch_size |
| "ValueError: invalid literal" | Config parsing | Check argument types |
| "FileNotFoundError" | Wrong path | Use absolute paths |
| "AssertionError: shape mismatch" | Architecture mismatch | Check model args |

## Debugging Tips

### Enable Debug Mode
```python
# Add at top of main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Intermediate Outputs
```bash
# After each epoch, check cached samples
python -c "
import pickle, os
cache_dir = 'cache/diffusion'
files = sorted(os.listdir(cache_dir))
print(f'Cached files: {files}')

latest = os.path.join(cache_dir, files[-1])
with open(latest, 'rb') as f:
    data = pickle.load(f)
    print(f'Keys: {data.keys()}')
    if 'final' in data:
        print(f'Final shape: {data[\"final\"].shape}')
"
```

### Profile Performance
```python
# Add to training loop
import time
start = time.time()
# ... training code ...
print(f'Epoch time: {time.time() - start:.2f}s')
```

## Still Having Issues?

If none of these solutions work:

1. Verify your Python version: `python --version` (need 3.8+)
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review the error message carefully
5. Check if similar issue in GitHub Issues
6. Create minimal reproducible example
