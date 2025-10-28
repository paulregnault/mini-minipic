"""Gateway for validation."""

import importlib
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

CMAKE_CACHE_FILENAME = "CMakeCache.txt"


def detect_setup(path: Path) -> str:
    cmake_cache_file = path / CMAKE_CACHE_FILENAME
    if not cmake_cache_file.exists():
        raise RuntimeError(f"Cannot find {cmake_cache_file}")

    cmake_cache_content = cmake_cache_file.read_text()
    matcher = re.findall(r"MINIPIC_SETUP:STRING=(.*)", cmake_cache_content)

    if not matcher:
        raise RuntimeError(f"Cannot find setup in {cmake_cache_file}")

    # return the first element, as we know there is at least one
    return matcher[0].strip()


def validate_setup(path, setup=None, threshold=1e-10):
    if not setup:
        setup = detect_setup(path)
        print(f"Autodetected setup: {setup}")

    module = importlib.import_module(f"libminipic.validation.{setup}", None)

    os.chdir(path)

    if not os.path.isdir("diags"):
        raise RuntimeError(
            "Directory diags should be present where you run this script"
        )

    module.validate(threshold)

    print(f"\033[32mBenchmark `{setup}` tested with success \033[39m")

    # force flush to see text on time
    # especially useful on HPC
    sys.stdout.flush()


def validate():
    parser = ArgumentParser(description="Run a validation script")

    parser.add_argument(
        "-s", "--setup", help="name of the setup to validate (default to autodetect)"
    )
    parser.add_argument(
        "-p",
        "--path",
        help="path to the execution directory (default to current working directory)",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--threshold", help="threshold for the validation", default=1e-10, type=float
    )

    args = parser.parse_args()

    validate_setup(args.path, args.setup, args.threshold)
