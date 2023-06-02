from time import time, process_time
from contextlib import contextmanager

@contextmanager
def timeit(name):
    start = time()
    start_cpu = process_time()
    try:
        yield
    finally:
        print(f"{name}: {time() - start:.3f} elapsed, {process_time() - start_cpu:.3f} CPU")
