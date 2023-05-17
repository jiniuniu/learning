import logger
from random import random
from time import sleep
from threading import Thread

log = logger.get_logger(__name__)


def task(number, threshold):
    value = random()
    sleep(value)
    if value < threshold:
        log.warning(f"线程-{number} 的执行时间: {value} 小于阈值 {threshold}.")
    log.info(f"线程-{number} 完成.")


def multi_task(num_tasks, threshold=0.2):
    threads = [
        Thread(target=task, args=(i, threshold)) for i in range(1, num_tasks + 1)
    ]
    # start threads
    for thread in threads:
        thread.start()
    # wait for threads to finish
    for thread in threads:
        thread.join()
