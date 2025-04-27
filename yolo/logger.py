import logging
import re
import os
import shutil
DEFAULT_LOG_LEVEL = -1

from yolo import ROOT_LOGGER_NAME
from yolo.colors import Colors

class ColoredFormatter(logging.Formatter):
    # Terminal color codes
        
    COLORS = {
        'DEBUG': Colors.LIGHT_BLUE,
        'INFO': Colors.LIGHT_GREEN,
        'WARNING': Colors.LIGHT_YELLOW,
        'ERROR': Colors.LIGHT_RED,
        'CRITICAL': Colors.NEGATIVE,
        'RESET': Colors.END,
        'GREY': Colors.FAINT,
        'YELLOW': Colors.YELLOW,
        'WHITE': Colors.END,
        'DARK_BLUE': Colors.BLUE
    }

    def format(self, record):
        """
        Format the log record.
        
        :param record: The log record to format.
        :type record: :class:`logging.LogRecord`
        """
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        time_color = self.COLORS['GREY']
        name_color = self.COLORS['YELLOW']
        level_color = log_color
        message_color = self.COLORS['WHITE']
        
        log_message = super().format(record)
        
        message = record.getMessage()
        message = re.sub(r'(localhost)', f"{self.COLORS['DARK_BLUE']}\\1{self.COLORS['RESET']}", message)
        message = re.sub(r'(port \d+)', f"{self.COLORS['DARK_BLUE']}\\1{self.COLORS['RESET']}", message)
        
        formatted_message = (f"{time_color}{self.formatTime(record, self.datefmt)}{self.COLORS['RESET']} - "
                             f"{name_color}{record.name}{self.COLORS['RESET']} - "
                             f"{level_color}{record.levelname}{self.COLORS['RESET']} - "
                             f"{message_color}{record.getMessage()}{self.COLORS['RESET']}")
        
        return formatted_message

def get_root_logger(name: str, level: int, verbose: bool) -> logging.Logger:
    """
    Create a new logger with the specified name and level.
    
    :param name: Name of the new logger
    :type name: str
    
    :param level: Level of the new logger
    :type level: int
    
    :param verbose: Whether to set the level to DEBUG
    :type verbose: bool
    
    :return: New logger
    :rtype: :class:`logging.Logger`
    """
    shutil.rmtree('logs', ignore_errors=True)
    
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if level == DEFAULT_LOG_LEVEL:
        if verbose:
            set_level = logging.DEBUG
        else:
            set_level = logging.WARNING
    else:
        set_level = level
    
    logger.setLevel(set_level)
    
    return logger

def create_log_directory_and_filename(name) -> tuple[str, str]:
    """
    Create a log directory and filename based on the name.
    
    :param name: The name of the logger.
    :type name: str
    
    :return: The log directory and filename.
    :rtype: tuple[str, str]
    """
    if '-' in name:
        log_dir = '-'.join(name.split('-')[0:2])
        filename = ''.join(name.split('-')[2:]) or name.split('-')[0]
    else:
        log_dir = filename = name
        
    log_dir = os.path.join('logs', log_dir)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir, filename

def get_logger(name: str) -> logging.Logger:
    """
    Create a new logger with the specified name and copy the level, 
    handlers, and formatter from the source logger.
    
    :param name: Name of the new logger
    :type name: str
    
    :return: New logger
    :rtype: :class:`logging.Logger`
    """
    
    # Create a new logger with the specified name
    new_logger = logging.getLogger(name)

    # Copy the level from the source logger
    source_logger = logging.getLogger(ROOT_LOGGER_NAME)
    new_logger.setLevel(source_logger.level)

    log_dir, filename = create_log_directory_and_filename(name)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{filename}.log"))
    
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add file handler to logger
    new_logger.addHandler(file_handler)

    # Copy the handlers from the source logger
    new_logger.handlers = [file_handler] + source_logger.handlers[:]

    return new_logger