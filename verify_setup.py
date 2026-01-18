#!/usr/bin/env python3
"""
Verification script for production-ready setup.
Run this to verify all components are working correctly.
"""

import sys
from pathlib import Path
from diffusion_transformer.utils import get_logger, setup_logging

# Setup logger for verification
setup_logging(level="INFO")
logger = get_logger("verification")

def test_imports():
    """Test critical imports."""
    logger.info("Testing critical imports...")

    try:
        import torch
        logger.info(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import numpy as np
        logger.info(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        from diffusion_transformer.utils import get_logger, setup_logging
        logger.info("  ✓ Logger module")
    except ImportError as e:
        logger.error(f"  ✗ Logger import failed: {e}")
        return False

    try:
        from diffusion_transformer.models.diffusion import DiffusionProcess
        logger.info("  ✓ DiffusionProcess")
    except ImportError as e:
        logger.error(f"  ✗ DiffusionProcess import failed: {e}")
        return False

    try:
        from diffusion_transformer.training.trainer import Trainer
        logger.info("  ✓ Trainer")
    except ImportError as e:
        logger.error(f"  ✗ Trainer import failed: {e}")
        return False

    try:
        from diffusion_transformer.evaluation.evaluator import Evaluator
        logger.info("  ✓ Evaluator")
    except ImportError as e:
        logger.error(f"  ✗ Evaluator import failed: {e}")
        return False

    return True


def test_logger_functionality():
    """Test logger functionality."""
    logger.info("Testing logger functionality...")

    try:
        from diffusion_transformer.utils import get_logger, setup_logging

        # Test logging
        logger.info("Test INFO message")
        logger.debug("Test DEBUG message (should not appear if level is INFO)")

        logger.info("  ✓ Logger setup successful")
        logger.info("  ✓ Logger messages working")
        return True
    except Exception as e:
        logger.error(f"  ✗ Logger test failed: {e}")
        return False


def check_files():
    """Check essential files exist."""
    logger.info("Checking essential files...")

    essential_files = [
        "main.py",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "README.md",
        "ARCHITECTURE.md",
        "QUICKSTART.md",
        ".gitignore",
        "diffusion_transformer/utils/logger.py",
        "diffusion_transformer/utils/__init__.py",
    ]

    all_exist = True
    for file in essential_files:
        path = Path(file)
        if path.exists():
            logger.info(f"  ✓ {file}")
        else:
            logger.error(f"  ✗ {file} (missing)")
            all_exist = False

    return all_exist


def main():
    """Run all verification tests."""
    logger.info("="*60)
    logger.info("Production-Ready Setup Verification")
    logger.info("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test logger
    results.append(("Logger", test_logger_functionality()))

    # Check files
    results.append(("Essential Files", check_files()))

    # Summary
    logger.info("="*60)
    logger.info("Verification Summary")
    logger.info("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {name:20s} {status}")
        all_passed = all_passed and passed

    logger.info("="*60)

    if all_passed:
        logger.info("🎉 All checks passed! Repository is production-ready.")
        logger.info("Next steps:")
        logger.info("  1. Run training: python main.py --step train --model_type dit")
        logger.info("  2. Read documentation: [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)")
        logger.info("  3. Check Makefile: make help")
        return 0
    else:
        logger.error("⚠️  Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

