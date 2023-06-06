import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor


thread_local = threading.local()


def get_session():
    # 在线程本地存储变量
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

# download a url and return the raw data, or None on error
def download_url(url):
    try:
        # open a connection to the server
        with urlopen(url, timeout=3) as connection:
            # read the contents of the html doc
            return connection.read()
    except:
        # bad url, socket timeout, http forbidden, etc.
        return None


def io_task(site):
    session = get_session()
    resp = session.get(site)
    return resp


def run_with_threads(n_jobs, sites):
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        executor.map(io_task, sites)

# URL列表
sites = ['https://docs.python.org/3/library/concurrency.html',
        'https://docs.python.org/3/library/concurrent.html',
        'https://docs.python.org/3/library/concurrent.futures.html',
        'https://docs.python.org/3/library/threading.html',
        'https://docs.python.org/3/library/multiprocessing.html',
        'https://docs.python.org/3/library/multiprocessing.shared_memory.html',
        'https://docs.python.org/3/library/subprocess.html',
        'https://docs.python.org/3/library/queue.html',
        'https://docs.python.org/3/library/sched.html',
        'https://docs.python.org/3/library/contextvars.html']


run_with_threads(n_jobs=10, sites=sites)


# SuperFastPython.com
# download document files and save to local files serially
from os import makedirs
from os.path import basename
from os.path import join
from urllib.request import urlopen
 
# download a url and return the raw data, or None on error
def download_url(url):
    try:
        # open a connection to the server
        with urlopen(url, timeout=3) as connection:
            # read the contents of the html doc
            return connection.read()
    except:
        # bad url, socket timeout, http forbidden, etc.
        return None
 
# save data to a local file
def save_file(url, data, path):
    # get the name of the file from the url
    filename = basename(url)
    # construct a local path for saving the file
    outpath = join(path, filename)
    # save to file
    with open(outpath, 'wb') as file:
        file.write(data)
    return outpath
 
# download and save a url as a local file
def download_and_save(url, path):
    # download the url
    data = download_url(url)
    # check for no data
    if data is None:
        print(f'>Error downloading {url}')
        return
    # save the data to a local file
    outpath = save_file(url, data, path)
    # report progress
    print(f'>Saved {url} to {outpath}')
 
# download a list of URLs to local files
def download_docs(urls, path):
    # create the local directory, if needed
    makedirs(path, exist_ok=True)
    # download each url and save as a local file
    for url in urls:
        download_and_save(url, path)
 
# python concurrency API docs
URLS = ['https://docs.python.org/3/library/concurrency.html',
        'https://docs.python.org/3/library/concurrent.html',
        'https://docs.python.org/3/library/concurrent.futures.html',
        'https://docs.python.org/3/library/threading.html',
        'https://docs.python.org/3/library/multiprocessing.html',
        'https://docs.python.org/3/library/multiprocessing.shared_memory.html',
        'https://docs.python.org/3/library/subprocess.html',
        'https://docs.python.org/3/library/queue.html',
        'https://docs.python.org/3/library/sched.html',
        'https://docs.python.org/3/library/contextvars.html']
# local path for saving the files
PATH = 'docs'
# download all docs
download_docs(URLS, PATH)