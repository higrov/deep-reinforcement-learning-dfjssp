#!/usr/bin/env python3
from typing import Any, Dict
import argparse

from utils.config import load_parameters
from solution_methods.runner import run_method


def main():
    parser = argparse.ArgumentParser(description="Run Gym Cooking experiment via TOML config.")
    parser.add_argument("-f", "--config", required=True, type=str, help="Path to TOML config file")
    args = parser.parse_args()

    params: Dict[str, Any] = load_parameters(args.config)

    # Expect the TOML to have env, method sections
    env_cfg = params.get("env", {})
    method_section = params.get("method", {})
    method_name = method_section.get("name")
    method_cfg = method_section.get("config", {})
    mode = params.get("mode", "eval")

    if method_name is None:
        raise ValueError("Config is missing [method].name")

    score, info = run_method(env_cfg, method_name, method_cfg, mode)
    print(f"Score: {score:.3f} | Info: {info}")


if __name__ == "__main__":
    main()
