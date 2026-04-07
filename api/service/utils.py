import logging
from enum import StrEnum

LOG_FORMAT = (
    "%(levelname)s:%(message)s:%(pathname)s:%(funcname)s:%(lineno)d"  # For debugs
)
LOG_FILE = "tranceition.log"


class LogLevels(StrEnum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    debug = "DEBUG"
    critical = "CRITICAL"


def setup_logging(log_level: str = LogLevels.error):
    log_level = str(log_level).upper()
    log_levels = [level.value for level in LogLevels]

    if log_level not in log_levels:
        logging.basicConfig(filename=LOG_FILE, encoding="utf-8", level=LogLevels.error)
        return

    if log_level == LogLevels.debug:
        logging.basicConfig(
            filename=LOG_FILE,
            encoding="utf-8",
            level=LogLevels.debug,
            format=LOG_FORMAT,
        )
        return

    logging.basicConfig(filename=LOG_FILE, encoding="utf-8", level=log_level)