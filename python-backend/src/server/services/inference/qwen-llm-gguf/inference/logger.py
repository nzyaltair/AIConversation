import io
import logging
import os
import sys

_logger = logging.getLogger("qwen-gguf")
_logger.setLevel(logging.INFO)

# 控制台输出
_wrapper = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_console_handler = logging.StreamHandler(_wrapper)
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
_logger.addHandler(_console_handler)

# 文件输出
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_file_handler = logging.FileHandler(os.path.join(_LOG_DIR, "latest.log"), encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
_logger.addHandler(_file_handler)


def info(msg):
    _logger.info(msg)


def warning(msg):
    _logger.warning(msg)


def error(msg):
    _logger.error(msg)


def debug(msg):
    _logger.debug(msg)


def set_level(level):
    _logger.setLevel(level)
