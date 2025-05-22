# At the top of the file, add or modify these imports
import logging
import os
from datetime import datetime

class DebugLogger:
    """A logger class for debugging purposes."""
    
    def __init__(self, name="debug"):
        self.logger = logging.getLogger(name)
        self.log_file = setup_logger()
    
    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)

def setup_logger():
    """Set up a file-based logger that can be used across parent and child processes."""
    # Create logs directory if it doesn't exist
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/retriever_{timestamp}.log"
    
    # Configure the logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'  # Append mode to allow multiple processes to write to the same file
    )
    
    # Also log to console for immediate feedback
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return log_file