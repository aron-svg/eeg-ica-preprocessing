import logging

logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set up logger
import logging_config  # noqa

logger = logging.getLogger("eeg_signal_preprocessing")
logger.setLevel(logging.INFO)
