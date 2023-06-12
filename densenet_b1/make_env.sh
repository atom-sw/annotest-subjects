#! /usr/bin/env bash

# Exiting when any command fails
set -e

<<<<<<< HEAD
# Create virtual environment
python3.6 -m venv env
source env/bin/activate
=======
# Parameters
BUG_ID="densenet_b1"

# Create virtual environment
conda create -n "$BUG_ID" python=3.6 -y
source "$ANACONDA3_DIRECTORY/etc/profile.d/conda.sh"
conda activate "$BUG_ID"
>>>>>>> dev
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
pip install pytest

# Remove keras.json file from home
<<<<<<< HEAD
rm ~/.keras/keras.json
=======
KERAS_CONF_JSON=~/.keras/keras.json
if [[ -f "$KERAS_CONF_JSON" ]]; then
    rm "$KERAS_CONF_JSON"
fi

>>>>>>> dev
