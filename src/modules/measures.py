from time import time
from colorama import Fore, Style

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"{Fore.BLUE}(Time Elapsed: {end_time - start_time:.4f}s){Style.RESET_ALL}")
        return result
    return wrapper
