#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="TF_PG_GANS_b1"
TEST_NAME_FILE_NAME="annotest_test_name.txt"

# Activate virtual environment
source "$ANACONDA3_DIRECTORY"/etc/profile.d/conda.sh
conda activate "$BUG_ID"

# Run manual test
python -m pip --version
TEST_NAME=$(<"$TEST_NAME_FILE_NAME")
echo "$TEST_NAME"

python -m pytest "$TEST_NAME"
