import logger
from modules.module import multi_task

log = logger.setup_logger(file_name="app_demo.log")


def main():
    log.debug(f"开始调用 multi_task 函数...")
    multi_task(num_tasks=5)
    log.debug("结束调用 multi_task 函数.")


if __name__ == "__main__":
    main()
