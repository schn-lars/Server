import logging
from enum import Enum

LOG_FORMAT = (
    "%(levelname)s:%(message)s:%(pathname)s:%(funcname)s:%(lineno)d"  # For debugs
)
LOG_FILE = "intenso.log"


class LogLevels(str, Enum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    debug = "DEBUG"
    critical = "CRITICAL"


def setup_logging(log_level: str = LogLevels.error):
    if isinstance(log_level, LogLevels):
        log_level = log_level.value

    log_level = log_level.upper()
    log_levels = [level.value for level in LogLevels]

    if log_level not in log_levels:
        logging.basicConfig(filename=LOG_FILE, encoding="utf-8", level=LogLevels.error.value)
        return

    if log_level == LogLevels.debug.value:
        logging.basicConfig(
            filename=LOG_FILE,
            encoding="utf-8",
            level=LogLevels.debug,
            format=LOG_FORMAT,
        )
        return

    logging.basicConfig(filename=LOG_FILE, encoding="utf-8", level=log_level)