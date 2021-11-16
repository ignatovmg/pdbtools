import sys
import traceback
import threading
import multiprocessing
from logging import FileHandler as FH

import logging
import logging.config

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'default': {
            'format': '[%(levelname)s] %(asctime)s %(funcName)s [pid %(process)d] - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'loggers': {
        'console': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        }
    }
}

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('console')


def get_file_handler(filename, mode='w', level='DEBUG'):
    h = logging.FileHandler(filename, mode=mode)
    h.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(funcName)s [pid %(process)d] - %(message)s'))
    h.setLevel(level)
    return h


def redirect_logger_to_file(logger, fname, mode='w', level='DEBUG'):
    new_handler = get_file_handler(fname, mode, level)
    old_handlers = logger.handlers
    logger.handlers = [new_handler]
    return old_handlers
