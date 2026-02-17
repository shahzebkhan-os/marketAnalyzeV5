import logging
import sys

def configure_logger():
    """
    Configures the root logger for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def get_logger(name):
    """
    Returns a standard library logger instance.
    """
    return logging.getLogger(name)
