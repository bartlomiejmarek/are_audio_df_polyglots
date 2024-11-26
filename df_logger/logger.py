import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


class Logger:
    def __init__(
            self,
            name: str,
            log_file: str = 'app.log',
            log_dir: str = './logs',
            max_bytes: int = 1000000,
            backup_count: int = 5,
            log_level: int = logging.INFO
    ):
        """
        Initialize the Logger object.
        """
        self.log_file = log_file
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = log_level
        self.logger: Optional[logging.Logger] = None
        self.name = name
        self.setup_logger()

    def setup_logger(self) -> None:
        """
        Set up the logger with rotation, formatting, and console output.
        """
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Ensure the log directory is secure
        os.makedirs(self.log_dir, exist_ok=True)
        os.chmod(self.log_dir, 0o700)

        log_path = os.path.join(self.log_dir, self.log_file)

        # Use a rotating file handler for file logging
        file_handler = RotatingFileHandler(log_path, maxBytes=self.max_bytes, backupCount=self.backup_count)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(self.log_level)

        # Use a stream handler for console logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(self.log_level)

        # Configure logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger.
        """
        return self.logger
