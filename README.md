# Diffusion Transformer for Football Event Sequence Generation

This repository contains a production-ready implementation of Diffusion Transformers (DiT) for generating realistic football event sequences. The project leverages latent diffusion on pre-computed event embeddings to capture complex patterns and long-range dependencies in football matches.

## Project Goals

Our aim is to develop a robust generative model capable of producing high-quality football event sequences that can be used for:
- **Synthetic Data Generation**: Creating realistic match scenarios for analysis and coaching.
- **Event Prediction**: Understanding the probabilistic nature of event sequences in different match contexts.
- **Performance Analysis**: Evaluating match events against a distribution of realistic alternatives.

## Key Features

- **Diffusion Transformer (DiT)**: A state-of-the-art transformer-based architecture for denoising latent sequences.
- **Latent Diffusion**: Operates in an optimized embedding space for efficient and high-quality generation.
- **Flexible Noise Schedules**: Support for linear, cosine, and quadratic schedules.
- **Fast Sampling**: Integrated DDIM sampling for accelerated inference.
- **Comprehensive Evaluation**: Modular evaluation suite including statistical, diversity, and realism metrics.
- **HPC Ready**: Optimized for cluster environments with Slurm support and mixed-precision training.

## Getting Started

For installation instructions and detailed usage examples, please refer to the [QUICKSTART.md](QUICKSTART.md) guide.

## Project Structure

```
football_events_generation/
├── main.py                          # Entry point for all tasks
├── diffusion_transformer/
│   ├── data/                       # Data loading and preprocessing logic
│   ├── models/                     # DiT architecture and diffusion process
│   ├── training/                   # Training loop and optimization
│   ├── evaluation/                 # Metrics and sample generation
│   ├── visualization/              # Results and progress visualization
│   └── utils/                      # Logging and shared utilities
├── cache/                          # Cached embeddings and samples
├── models/                         # Model checkpoints and saved states
├── outputs/                        # Visualizations and evaluation results
└── csv/                           # Input event data (CSV format)
```

## Citation

```bibtex
@software{diffusion_transformer_2025,
  author = {Ben-Tzvi, Amit and Zendel, Ilan},
  title = {Diffusion Transformer for Football Event Sequence Generation},
  year = {2025},
  url = {https://github.com/Amit-bt-technion/football_events_generation}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- **Amit Ben-Tzvi** - (amit-ben@campus.technion.ac.il)
- **Ilan Zendel** - (ilan.zendel@campus.technion.ac.il)
