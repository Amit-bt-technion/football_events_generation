#!/usr/bin/env python3
"""
Verification script for production-ready setup.
Run this to verify all components are working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test critical imports."""
    print("Testing critical imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        from diffusion_transformer.utils import get_logger, setup_logging
        print("  ✓ Logger module")
    except ImportError as e:
        print(f"  ✗ Logger import failed: {e}")
        return False

    try:
        from diffusion_transformer.models.diffusion import DiffusionProcess
        print("  ✓ DiffusionProcess")
    except ImportError as e:
        print(f"  ✗ DiffusionProcess import failed: {e}")
        return False

    try:
        from diffusion_transformer.training.trainer import Trainer
        print("  ✓ Trainer")
    except ImportError as e:
        print(f"  ✗ Trainer import failed: {e}")
        return False

    try:
        from diffusion_transformer.evaluation.evaluator import Evaluator
        print("  ✓ Evaluator")
    except ImportError as e:
        print(f"  ✗ Evaluator import failed: {e}")
        return False

    return True


def test_logger():
    """Test logger functionality."""
    print("\nTesting logger functionality...")

    try:
        from diffusion_transformer.utils import get_logger, setup_logging

        # Setup logging
        setup_logging(level="INFO")
        logger = get_logger("test")

        # Test logging
        logger.info("Test INFO message")
        logger.debug("Test DEBUG message (should not appear)")

        print("  ✓ Logger setup successful")
        print("  ✓ Logger messages working")
        return True
    except Exception as e:
        print(f"  ✗ Logger test failed: {e}")
        return False


def check_files():
    """Check essential files exist."""
    print("\nChecking essential files...")

    essential_files = [
        "main.py",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "README.md",
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        ".gitignore",
        "diffusion_transformer/utils/logger.py",
        "diffusion_transformer/utils/__init__.py",
    ]

    all_exist = True
    for file in essential_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            all_exist = False

    return all_exist


def main():
    """Run all verification tests."""
    print("="*60)
    print("Production-Ready Setup Verification")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test logger
    results.append(("Logger", test_logger()))

    # Check files
    results.append(("Essential Files", check_files()))

    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")
        all_passed = all_passed and passed

    print("="*60)

    if all_passed:
        print("\n🎉 All checks passed! Repository is production-ready.")
        print("\nNext steps:")
        print("  1. Run training: python main.py --step train --model_type dit")
        print("  2. Read documentation: less README.md")
        print("  3. Check Makefile: make help")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

