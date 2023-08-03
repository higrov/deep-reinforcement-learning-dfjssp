import glob
import os

import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def get_folder_name(arglist, level="", suffix=""):
    folder = f"{level}/agents-{arglist.num_agents}/orders-{arglist.num_orders}/"
    model = ""
    if arglist.model1 is not None:
        model += f"model1-{arglist.model1}"
    if arglist.model2 is not None:
        model += f"_model2-{arglist.model2}"
    if arglist.model3 is not None:
        model += f"_model3-{arglist.model3}"
    if arglist.model4 is not None:
        model += f"_model4-{arglist.model4}"
    if arglist.model5 is not None:
        model += f"_model5-{arglist.model5}"
    folder += model

    return folder + suffix


def _squash_info(info):
    toDiscard = {"t", "rep", "recording", "TimeLimit.truncated", "obs", "termination_info", "done", "terminal_observation", "termination_stats", "terminal_state"}
    keys = set([k for i in info for k in i.keys() if k not in toDiscard])
    new_info = {}
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean

    return new_info
