import os
from datetime import datetime
from pathlib import Path

import wandb


class ExperimentTracker:
    """
    A context manager for tracking machine learning experiments using Weights & Biases (wandb).
    This class provides a convenient way to initialize and manage wandb runs with automatic
    run naming based on configuration files and timestamps.
    Args:
        project_name (str): Name of the wandb project to log experiments to.
        config (dict, optional): Configuration dictionary to log with the experiment.
            Defaults to empty dict if None.
        log_dir (str, optional): Directory path where experiment logs will be stored.
            Defaults to "experiment_logs".
        config_file (str, optional): Path to configuration file. Used to generate
            meaningful run names. Defaults to None.
    Attributes:
        project_name (str): The wandb project name.
        config (dict): Configuration parameters for the experiment.
        log_dir (str): Directory for storing logs.
        config_file (str): Path to the configuration file.
        run: The wandb run object (None until context is entered).
    Example:
        >>> with ExperimentTracker("my-project", config={"lr": 0.001}) as run:
        ...     run.log({"loss": 0.5})
        >>> # With config file
        >>> tracker = ExperimentTracker(
        ...     project_name="nlp-experiments",
        ...     config_file="configs/bert_config.yaml"
        ... )
        >>> with tracker as run:
        ...     run.log({"accuracy": 0.95})
    Note:
        Run names are automatically generated in the format: "{config_name}-{timestamp}"
        where timestamp follows the pattern "dd-mmm-yyyy-hh:mmam/pm".
    """

    def __init__(
        self, project_name, config=None, log_dir="experiment_logs", config_file=None
    ):
        self.project_name = project_name
        self.config = config or {}
        self.log_dir = log_dir
        self.config_file = config_file
        self.run = None

    def __enter__(self):
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%d-%b-%Y-%I:%M%p").lower()

        if self.config_file:
            config_name = Path(self.config_file).stem  # Get filename without extension
        else:
            config_name = "default"

        run_name = f"{config_name}-{timestamp}"

        self.run = wandb.init(
            project=self.project_name,
            config=self.config,
            dir=self.log_dir,
            name=run_name,
        )
        return self.run

    def __exit__(self, *args):
        wandb.finish()
