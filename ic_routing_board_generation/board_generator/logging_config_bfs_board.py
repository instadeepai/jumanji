# A file that contains the logging configuration for the BFS board generator.
# Path: ic_routing_board_generation/board_generator/logging_config_bfs_board.py
import logging
import logging.config

# Debug - 10
# Info - 20
# Warning - 30
# Error - 40
# Critical - 50


# Define the logging configuration with specific settings for the BFS board generator and a specified file to write to.
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'bfs_board.log',
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
}

# Configure the logging
logging.config.dictConfig(LOGGING_CONFIG)

# LOGGING_CONFIG = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'formatters': {
#         'standard': {
#             'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
#         },
#     },
#     'handlers': {
#         'default': {
#             'level': 'DEBUG',
#             'class': 'logging.StreamHandler',
#             'formatter': 'standard'
#         },
#     },
#     'loggers': {
#         '': {
#             'handlers': ['default'],
#             'level': 'DEBUG',
#             'propagate': True
#         },
#     }
# }
#
# # Configure the logging
# logging.config.dictConfig(LOGGING_CONFIG)
