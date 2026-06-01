"""
Test suite for the transformer model using pytest.

This module contains tests for verifying the correctness and functionality
of the transformer model implementation with beautiful output and comprehensive reporting.
"""

import logging
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config import Config
from loss import get_loss_function
from model import build_transformer
from train import train_batch_CE
from utils import BatchTensors, get_dataloader

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize rich console for beautiful output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestTransformerModel:
    """Test cases for the transformer model using pytest."""

    @pytest.fixture(scope="class")
    def config_file(self):
        """Fixture for configuration file."""
        config_path = Path("config.toml")
        if not config_path.exists():
            pytest.skip(f"Configuration file {config_path} not found")
        return config_path

    @pytest.fixture(scope="class")
    def config(self, config_file):
        """Fixture for loaded configuration."""
        with console.status("[bold green]Loading configuration..."):
            config = Config(config_file=config_file)
        console.print("✅ Configuration loaded successfully", style="bold green")
        return config

    @pytest.fixture(scope="class")
    def transformer_and_dataset(self, config):
        """Fixture for transformer model and dataset."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Building transformer model and dataset...", total=None
            )

            start_time = time.time()
            transformer, ds = build_transformer(config)
            build_time = time.time() - start_time

            progress.update(
                task, completed=True, description=f"✅ Built in {build_time:.2f}s"
            )

        # Display model info in a nice table
        table = Table(title="🤖 Model Information")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        param_count = sum(p.numel() for p in transformer.parameters())
        table.add_row("Total Parameters", f"{param_count:,}")
        table.add_row("Device", str(transformer.device))
        table.add_row("Hidden Size", str(config.model.hidden_size))
        table.add_row("Number of Layers", str(config.model.n_layers))
        table.add_row("Number of Heads", str(config.model.n_heads))

        console.print(table)

        return transformer, ds

    @pytest.fixture
    def dataloader(self, transformer_and_dataset, config):
        """Fixture for data loader."""
        _, ds = transformer_and_dataset
        with console.status("[bold blue]Creating data loader..."):
            train_data_loader = get_dataloader(ds, config)
        console.print("✅ Data loader created", style="bold green")
        return train_data_loader

    @pytest.mark.unit
    def test_config_loading(self, config):
        """Test that configuration is loaded correctly."""
        console.print(Panel.fit("⚙️ Testing Configuration Loading", style="bold blue"))

        # Verify essential config attributes exist
        sections = ["model", "training", "dataset", "loss"]
        for section in sections:
            assert hasattr(config, section), f"Config should have {section} section"

        # Create configuration summary table
        table = Table(title="📋 Configuration Summary")
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Property", style="green")
        table.add_column("Value", style="yellow")

        # Model config
        model_config = config.model
        table.add_row("Model", "Hidden Size", str(model_config.hidden_size))
        table.add_row("", "Max Sequence Length", str(model_config.max_seq_len))
        table.add_row("", "Vocabulary Size", str(model_config.vocab_size))
        table.add_row("", "Number of Layers", str(model_config.n_layers))
        table.add_row("", "Number of Heads", str(model_config.n_heads))

        # Training config
        training_config = config.training
        table.add_row("Training", "Batch Size", str(training_config.batch_size))
        table.add_row("", "Learning Rate", str(training_config.learning_rate))
        table.add_row("", "Device", str(training_config.device))

        # Loss config
        table.add_row("Loss", "Type", str(config.loss.type))
        table.add_row("", "Label Smoothing", str(config.loss.label_smoothing))

        console.print(table)

        # Verify key values
        assert model_config.hidden_size > 0, "Hidden size should be positive"
        assert model_config.max_seq_len > 0, "Max sequence length should be positive"
        assert training_config.batch_size > 0, "Batch size should be positive"

        console.print("✅ Configuration validation completed", style="bold green")


# Standalone test function with rich output (not a pytest test)
def standalone_test_run(config_file: Path):
    """
    Standalone test function for transformer forward pass with beautiful output.
    """
    console.print(Panel.fit("🔥 Standalone Transformer Test", style="bold red"))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Load config
            task = progress.add_task("Loading configuration...", total=None)
            config = Config(config_file=config_file)
            progress.update(task, completed=True, description="✅ Configuration loaded")

            # Build model
            task = progress.add_task("Building transformer model...", total=None)
            transformer, ds = build_transformer(config)
            progress.update(task, completed=True, description="✅ Model built")

            # Create dataloader
            task = progress.add_task("Creating data loader...", total=None)
            train_data_loader = get_dataloader(ds, config)
            progress.update(task, completed=True, description="✅ Data loader created")

            # Get batch
            task = progress.add_task("Loading batch...", total=None)
            batch = next(iter(train_data_loader))
            progress.update(task, completed=True, description="✅ Batch loaded")

            # Move to device
            task = progress.add_task("Moving to device...", total=None)
            batch_on_device = BatchTensors(
                src_batch_X=batch.src_batch_X.to(transformer.device),
                tgt_batch_X=batch.tgt_batch_X.to(transformer.device),
                tgt_batch_y=batch.tgt_batch_y.to(transformer.device),
                src_batch_X_pad_mask=batch.src_batch_X_pad_mask.to(transformer.device),
                tgt_batch_X_pad_mask=batch.tgt_batch_X_pad_mask.to(transformer.device),
            )
            progress.update(task, completed=True, description="✅ Moved to device")

            # Forward pass
            task = progress.add_task("Executing forward pass...", total=None)
            with torch.no_grad():
                loss_fn = get_loss_function(config, pad_id=ds.tokenizer.pad_id)
                if config.loss.type.lower() == "cross_entropy":
                    loss = train_batch_CE(
                        model=transformer, batch=batch_on_device, loss_fn=loss_fn
                    )
                    progress.update(
                        task, completed=True, description="✅ Forward pass completed"
                    )
                else:
                    raise ValueError(f"Unsupported loss type: {config.loss.type}")

        # Success panel
        success_info = [
            f"Loss: {loss.item():.5f}",
            f"Device: {transformer.device}",
        ]

        # Try to get parameter count if available
        try:
            if hasattr(transformer, "parameters"):
                param_count = sum(p.numel() for p in transformer.parameters())
                success_info.append(f"Parameters: {param_count:,}")
            elif hasattr(transformer, "named_parameters"):
                param_count = sum(
                    p.numel() for name, p in transformer.named_parameters()
                )
                success_info.append(f"Parameters: {param_count:,}")
        except Exception:  # pylint: disable=broad-except
            success_info.append("Parameters: Unable to count")

        console.print(
            Panel("\n".join(success_info), title="🎉 Test Successful", style="green")
        )

    except Exception as e:  # pylint: disable=broad-except
        console.print(Panel(f"Error: {e!s}", title="💥 Test Failed", style="red"))
        raise


if __name__ == "__main__":
    # Run with rich output
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])
