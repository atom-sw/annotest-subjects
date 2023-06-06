#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Activate virtual environment
source env/bin/activate

# Run manual test
python -m pytest tests_manual/test_b1.py::test_b1
