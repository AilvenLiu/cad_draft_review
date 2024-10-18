import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('cad_draft_review')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'cad_draft_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_format = logging.Formatter(format_string)
    file_format = logging.Formatter(format_string)
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create and configure logger
logger = setup_logging()