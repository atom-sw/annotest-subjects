#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="Visual_Semantic_Alignments_b2"

# Create virtual environment
conda create -n "$BUG_ID" python=3.6 -y
source "$ANACONDA3_DIRECTORY/etc/profile.d/conda.sh"
conda activate "$BUG_ID"
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt --ignore-installed certifi
pip install pytest

# Remove keras.json file from home
KERAS_CONF_JSON=~/.keras/keras.json
if [[ -f "$KERAS_CONF_JSON" ]]; then
    rm "$KERAS_CONF_JSON"
fi

# Create keras.json file for this bug
cat > "$KERAS_CONF_JSON" <<- EOM
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_data_format": "channels_last"
}
EOM
