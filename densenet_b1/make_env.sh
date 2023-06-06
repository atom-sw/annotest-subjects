#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Create virtual environment
python3.6 -m venv env
source env/bin/activate
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
pip install pytest

# Remove keras.json file from home
rm ~/.keras/keras.json
