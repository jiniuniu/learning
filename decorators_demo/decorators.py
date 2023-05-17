import random
import time
from functools import wraps, lru_cache
from ratelimit import limits, sleep_and_retry
import requests
from dataclasses import dataclass


def logger(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        """wrapper documentation"""
        print(f"----- {function.__name__}: start -----")
        output = function(*args, **kwargs)
        print(f"----- {function.__name__}: end -----")
        return output

    return wrapper


@logger
def add_two_numbers(a, b):
    """this function adds two numbers"""
    return a + b


add_two_numbers.__name__
# 'add_two_numbers'

add_two_numbers.__doc__
# 'this function adds two numbers'

####################################


@lru_cache(maxsize=None)
def heavy_duty(n):
    sleep_time = n + random.random()
    time.sleep(sleep_time)


def cache(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key in wrapper.cache:
            output = wrapper.cache[cache_key]
        else:
            output = function(*args)
            wrapper.cache[cache_key] = output
        return output

    wrapper.cache = dict()
    return wrapper


# 第一次调用
# %%time
heavy_duty(0)
# CPU times: user 1.18 ms, sys: 1.74 ms, total: 2.93 ms
# Wall time: 854 ms

# 第二次调用
# %%time
heavy_duty(0)
# CPU times: user 4 µs, sys: 0 ns, total: 4 µs
# Wall time: 8.11 µs


################################################################


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 函数调用耗时：{end - start:.6f}")
        return result

    return wrapper


@timeit
def long_running_task():
    time.sleep(1)


long_running_task()
# long_running_task 函数调用耗时：1.001168

########################################################################


def retry(num_retries, exception_to_check, sleep_time=0):
    """
    Decorator that retries the execution of a function if it raises a specific exception.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(1, num_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    print(f"{func.__name__} raised {e.__class__.__name__}. Retrying...")
                    if i < num_retries:
                        time.sleep(sleep_time)
            raise e

        return wrapper

    return decorate


@retry(num_retries=3, exception_to_check=ValueError, sleep_time=1)
def random_value():
    value = random.randint(1, 5)
    if value == 3:
        raise ValueError("Value cannot be 3")
    return value


random_value()
# random_value raised ValueError. Retrying...
# 1

random_value()
# 5

################################################################


def countcall(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        result = func(*args, **kwargs)
        print(f"{func.__name__} has been called {wrapper.count} times")
        return result

    wrapper.count = 0
    return wrapper


@countcall
def process_data():
    pass


process_data()
# process_data has been called 1 times
process_data()
# process_data has been called 2 times
process_data()
# process_data has been called 3 times

############################################################################


def rate_limited(max_per_second):
    min_interval = 1.0 / float(max_per_second)

    def decorate(func):
        last_time_called = 0.0

        @wraps(func)
        def rate_limited_function(*args, **kargs):
            elapsed = time.perf_counter() - last_time_called
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kargs)
            last_time_called = time.perf_counter()
            return ret

        return rate_limited_function

    return decorate


################################################################
FIFTEEN_MINUTES = 900


@sleep_and_retry
@limits(calls=15, period=FIFTEEN_MINUTES)
def call_api(url):
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("API response: {}".format(response.status_code))
    return response


################################################################
@dataclass
class Person:
    name: str
    age: int

    def __eq__(self, other):
        if isinstance(other, Person):
            return self.age == other.age
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Person):
            return self.age < other.age
        return NotImplemented


person_a = Person(name="张三", age=30)

person_b = Person(name="李四", age=40)

print(person_a)
# Person(name='张三', age=30)

print(person_a < person_b)
# True


################################################################
class Registry(object):
    def __init__(self):
        self._functions = []

    def register(self, decorated):
        self._functions.append(decorated)
        return decorated

    def run_all(self, *args, **kwargs):
        return_values = []
        for func in self._functions:
            return_values.append(func(*args, **kwargs))
        return return_values


r = Registry()


@r.register
def return_five():
    return 5


@r.register
def return_three():
    return 3


print(r.run_all())
# [5, 3]
