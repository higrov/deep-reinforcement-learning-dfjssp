

from asyncio.log import logger
from functools import wraps
import logging
import time
import random
import numpy as np

logger = logging.getLogger(__name__)


def agent_settings(arglist, agent_name):
    if agent_name[-1] == "1": return arglist.model1, arglist.model1_path
    elif agent_name[-1] == "2": return arglist.model2, arglist.model2_path
    elif agent_name[-1] == "3": return arglist.model3, arglist.model3_path
    elif agent_name[-1] == "4": return arglist.model4, arglist.model4_path
    elif agent_name[-1] == "5": return arglist.model5, arglist.model5_path
    else: raise ValueError("Agent name doesn't follow the right naming, `agent-<int>`")

def find_key_in(ttuple, key):
    # check if key is in tuple
    for i, entry in enumerate(ttuple):
        if key in entry:
            return i

    return -1

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)