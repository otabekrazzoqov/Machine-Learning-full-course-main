# src/logger.py

import logging
import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)                       # Creating logs/ directory if it doesn't exist

    log_filename = os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y-%m-%d')}.log"
    )                                                          # Log file named by date

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)                             # capture everything

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)                      # less verbose in terminal
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger