import queue
import requests
import threading
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

thread_local = threading.local()


def get_session():
    # Get a different session for each thread, because `Session` is not thread-safe.
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


class MultiThreadingPubSub:
    def __init__(self, num_jobs=10):
        self.queue = queue.Queue(maxsize=num_jobs)
        self.num_jobs = num_jobs
        self.event = threading.Event()

    def publisher(self, url):
        session = get_session()
        resp = session.get(url)
        result = {"site": url, "status": resp.status_code}
        self.queue.put(result)

    def subscriber(self):
        while not self.event.is_set() or not self.queue.empty():
            try:
                result = self.queue.get(timeout=1)
                print(f"message received: {result}")
            except queue.Empty:
                continue

    def run(self, urls: List[str]):
        futures = []

        with ThreadPoolExecutor(max_workers=self.num_jobs) as executor:
            future_sub = executor.submit(self.subscriber)
            futures.append(future_sub)

            for url in urls:
                future_pub = executor.submit(self.publisher, url)
                futures.append(future_pub)

            time.sleep(2)
            self.event.set()

        futures_completed = as_completed(futures)
        try:
            for future in futures_completed:
                future.result()
        except Exception as e:
            print(f"An error occured in a thread: {e}")


if __name__ == "__main__":
    URLS = [
        "https://docs.python.org/3/library/concurrency.html",
        "https://docs.python.org/3/library/concurrent.html",
        "https://docs.python.org/3/library/concurrent.futures.html",
        "https://docs.python.org/3/library/threading.html",
        "https://docs.python.org/3/library/multiprocessing.html",
        "https://docs.python.org/3/library/multiprocessing.shared_memory.html",
        "https://docs.python.org/3/library/subprocess.html",
        "https://docs.python.org/3/library/queue.html",
        "https://docs.python.org/3/library/sched.html",
        "https://docs.python.org/3/library/contextvars.html",
    ]
    NUM_JOBS = 10
    pub_sub = MultiThreadingPubSub(NUM_JOBS)
    pub_sub.run(URLS)


################################ 运行结果 #################################
# message received: {'site': 'https://docs.python.org/3/library/sched.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/concurrent.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/multiprocessing.shared_memory.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/threading.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/concurrent.futures.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/contextvars.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/queue.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/concurrency.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/subprocess.html', 'status': 200}
# message received: {'site': 'https://docs.python.org/3/library/multiprocessing.html', 'status': 200}
