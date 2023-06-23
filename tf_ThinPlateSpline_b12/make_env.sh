#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="tf_ThinPlateSpline_b12"

# Create virtual environment
conda create -n "$BUG_ID" python=3.6 -y
source "$ANACONDA3_DIRECTORY/etc/profile.d/conda.sh"
conda activate "$BUG_ID"
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
pip install pytest

