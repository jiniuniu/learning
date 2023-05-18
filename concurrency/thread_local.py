import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor


thread_local = threading.local()


def get_session():
    # 线程本地存储变量
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def io_task(site):
    session = get_session()
    resp = session.get(site)


def run_with_threads(n_jobs, sites):
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        executor.map(io_task, sites)


sites = [...]
run_with_threads(n_jobs=10, sites=sites)
