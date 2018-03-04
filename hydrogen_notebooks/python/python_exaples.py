# %%
import time
from datetime import datetime
import os

int(time.time())
datetime.strptime('2000-01-03', '%Y-%m-%d').timestamp()

# %%

def decorator_with_args(a, b):
    def decorator(process):
        print(f"decorator outer_args: {a}, {b}")

        def wrapper():
            print(f"wrapper outer_args: {a}, {b}")
            return process(10)
        return wrapper

    return decorator


@decorator_with_args(1, 6)
def process_args(data):
    print(f"process_args: {data}")


process_args()


def decorator_with_kwargs(**kwargs):
    def decorator(process):
        print(f"kwargs decorator outer_args: {kwargs}")

        def wrapper():
            print(f"kargs wrapper outer_args: {kwargs}")
            return process(10)
        return wrapper

    return decorator


@decorator_with_kwargs(path='/root/test', value='testing')
def process_kwargs(data):
    print(f"data: {data}")


process_kwargs()

# %%
file_name = os.path.join(os.environ['HOME'], '.1forge', 'apikey')
with open(file_name, 'r') as file:
    apikey = file.read()
print(apikey.rstrip())
