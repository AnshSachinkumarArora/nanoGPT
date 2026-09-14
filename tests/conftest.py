import torch
import pytest

def pytest_configure(config):
    if not torch.cuda.is_available():
        pytest.exit("Aborting suite: CUDA is required but not available.", returncode=1)