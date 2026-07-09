#!/usr/bin/env python
import logging
import logging.config
import os
import sys

import yaml

default_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)8s |  %(funcName)s | %(message)s",
    "%Y-%m-%d %H:%M:%S",
)


class ColoredFormatter(logging.Formatter):
    """Formatter that colorizes log lines based on level (console only)."""

    COLORS = {
        logging.DEBUG: "\x1b[38;20m",    # gris
        logging.INFO: "\x1b[36;20m",     # cyan
        logging.WARNING: "\x1b[33;20m",  # jaune
        logging.ERROR: "\x1b[31;20m",    # rouge
        logging.CRITICAL: "\x1b[31;1m",  # rouge gras
    }
    RESET = "\x1b[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{super().format(record)}{self.RESET}"

__logger = logging.getLogger(__name__)

__ch = logging.StreamHandler(sys.stdout)
__ch.setFormatter(default_formatter)

# default behavior
__logger.addHandler(__ch)

# create logs/ folder
if not os.path.exists("logs"):
    os.makedirs("logs")

# Parse config file
__logger_config_file = os.path.join(os.path.dirname(__file__), "logger_config.yaml")
if os.path.exists(__logger_config_file) and os.path.isfile(
    __logger_config_file
):
    try:
        log_cfg = yaml.safe_load(open(__logger_config_file))
        logging.config.dictConfig(log_cfg)   # NOSONAR

    except Exception as ex:
        __logger.error(
            f"Unexpected error whilst reading logger configuration file {__logger_config_file}"
        )
        __logger.error(f"Exception: {ex}")
        raise
else:
    __logger.warning(
        f"no configuration file {__logger_config_file} found, using default setting"
    )
