#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="TF_PG_GANS_b2"

# Create virtual environment
conda create -n "$BUG_ID" python=3.6 -y
source "$ANACONDA3_DIRECTORY/etc/profile.d/conda.sh"
conda activate "$BUG_ID"
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
pip install pytest

# Remove keras.json file from home
KERAS_CONF_JSON=~/.keras/keras.json
if [[ -f "$KERAS_CONF_JSON" ]]; then
    rm "$KERAS_CONF_JSON"
fi
