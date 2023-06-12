#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="keras_frcnn_b1"

# Activate virtual environment
source "$ANACONDA3_DIRECTORY"/etc/profile.d/conda.sh
conda activate "$BUG_ID"

# Run manual test
python -m pip --version

python -m pytest tests_manual/test_failing.py::test_failing
