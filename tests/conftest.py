"""
Pytest configuration and fixtures.

This file contains global pytest configuration and shared fixtures.
"""

import logging

import pytest
import torch
from rich.console import Console

# Initialize console for rich output
console = Console()


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add skip marker for GPU tests if no GPU is available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    console.print("🔥 Starting Transformer Test Suite", style="bold red")
    yield
    console.print("✅ Test Suite Completed", style="bold green")


@pytest.fixture
def suppress_logs(caplog):
    """Fixture to suppress logs during testing."""
    caplog.set_level(logging.WARNING)
    return caplog
