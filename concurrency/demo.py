import time
import threading


def io_task(id, num):
    start_time = time.time()
    print(f"IO Task-{id} started.")
    for _ in range(num):
        # 模拟请求响应的等待
        time.sleep(3)
    duration = time.time() - start_time
    print(f"IO Task-{id} finished in {duration:.2f} seconds.")


def cpu_task(id, num):
    start_time = time.time()
    print(f"CPU Task-{id} started.")
    sum = 0
    # 模拟 CPU-heavy job
    for i in range(num):
        sum += i
    duration = time.time() - start_time
    print(f"CPU Task-{id} finished in {duration:.2f} seconds.")


def run_with_threads(n_jobs, num, task):
    threads = []
    for id in range(n_jobs):
        thread = threading.Thread(target=task, args=(id, num))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


start_one_thread = time.time()
# Run with a single thread, with a big number.
run_with_threads(n_jobs=1, num=3, task=io_task)
duration = time.time() - start_one_thread
print(f"job finished in {duration:.2f} seconds with a single thread.")

start_three_threads = time.time()
# Run with three threads with a smaller number. The total number of three threads
# adds up to the one of a single thread so the result is comparable.
run_with_threads(n_jobs=3, num=1, task=io_task)
duration = time.time() - start_three_threads
print(f"job finished in {duration:.2f} seconds with three threads.")


############################ 执行结果 ###############################
# IO Task-0 started.
# IO Task-0 finished in 9.01 seconds.
# job finished in 9.01 seconds with a single thread.
# IO Task-0 started.
# IO Task-1 started.
# IO Task-2 started.
# IO Task-0 finished in 3.00 seconds.
# IO Task-1 finished in 3.01 seconds.
# IO Task-2 finished in 3.00 seconds.
# job finished in 3.01 seconds with three threads.
