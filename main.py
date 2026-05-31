import click
from pathlib import Path
import logging
from log_config import setup_log_formatting
from run import run_train


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the configuration file",
)
@click.option(
    "--notebook",
    is_flag=True,
    default=False,
    help="Use raw text logging for Jupyter notebooks",
)
def main(config: Path, notebook: bool):
    """
    Main entry point for transformer training with configurable parameters.

    Args:
        config (Path): Path to the TOML configuration file.
        notebook (bool): If set, uses raw text logging suitable for Jupyter.
    """
    setup_log_formatting(for_notebook=notebook)
    logging.info(f"Running transformer with config {config}")
    run_train(config_file=config)


if __name__ == "__main__":
    main()
