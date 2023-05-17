import logging
import sys

# from logger.handlers import KafkaHandler

APP_LOGGER_NAME = "logging_demo"


def setup_logger(logger_name=APP_LOGGER_NAME, **kwargs):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)

    if "file_name" in kwargs:
        file_name = kwargs["file_name"]
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if "kafka" in kwargs:
        params = kwargs["kafka"]
        hosts = params["hosts"]
        topic = params["topic"]
        kh = KafkaHandler(hostlist=hosts, topic=topic)
        kh.setFormatter(formatter)
        logger.addHandler(kh)

    return logger


def get_logger(module_name):
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)
