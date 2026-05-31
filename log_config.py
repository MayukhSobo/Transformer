import logging


class LogColors:
    BLUE = "\033[94m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """
    Injects ANSI color codes into log records
    based on severity (INFO, WARNING, ERROR).
    """

    def format(self, record: logging.LogRecord) -> str:
        color_map = {
            logging.INFO: LogColors.BLUE,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
        }
        color = color_map.get(record.levelno, "")
        # Inject fields used by the format string
        record.color_on = color
        record.color_off = LogColors.RESET
        return super().format(record)


def setup_log_formatting(for_notebook: bool = False, level: int = logging.INFO) -> None:
    """
    Configures the root logger's handler and formatter.

    Clears any existing handlers to prevent duplicate output, then attaches a
    StreamHandler with a colored formatter (plain text in notebook environments).

    :param for_notebook: Use a plain formatter without ANSI color codes (True for
        Jupyter notebooks, False for terminal output).
    :param level: Minimum log level for the root logger. Defaults to logging.INFO.
    :return: None
    """
    root_logger = logging.getLogger()

    # Clear existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    if for_notebook:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    else:
        formatter = ColoredFormatter(
            "%(color_on)s%(asctime)s - %(levelname)s - %(message)s%(color_off)s"
        )
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
