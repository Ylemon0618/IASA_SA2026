from time import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Time Elapsed: {end_time - start_time:.4f}s")
        return result
    return wrapper
