import logging

logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set up logger
import logging_config  # noqa

logger = logging.getLogger("python_template")
logger.setLevel(logging.INFO)
