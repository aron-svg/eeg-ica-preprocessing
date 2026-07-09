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
