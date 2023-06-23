#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="BiDAF_PyTorch_b1"
DATA_DIRECTORY_PATH=~/annotest_subjects_data

# File id on Google drive
FILE_ID=1-SzXgZCGulUB25ngwODASSvt33Ut1SEf

BUG_ID_DATA_DIRECTORY_PATH=$DATA_DIRECTORY_PATH/$BUG_ID
FILE_PATH=$BUG_ID_DATA_DIRECTORY_PATH/$BUG_ID.zip

if [ -d "$BUG_ID_DATA_DIRECTORY_PATH" ]; then
  rm -rf $BUG_ID_DATA_DIRECTORY_PATH
fi

mkdir -p $BUG_ID_DATA_DIRECTORY_PATH

wget --load-cookies /tmp/cookies.txt \
 "https://docs.google.com/uc?export=download&confirm=$(wget \
  --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
   --no-check-certificate \
   'https://docs.google.com/uc?export=download&id=$FILE_ID' -O- \
    | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" \
    -O $FILE_PATH && rm -rf /tmp/cookies.txt

unzip $FILE_PATH -d $BUG_ID_DATA_DIRECTORY_PATH
rm $FILE_PATH

# --------- Script from the project ---------

SQUAD_DIR=$HOME/data/squad
GLOVE_DIR=$HOME/data/glove

if [ -d "$SQUAD_DIR" ]; then
  rm -rf "$SQUAD_DIR"
fi

if [ -d "$GLOVE_DIR" ]; then
  rm -rf "$GLOVE_DIR"
fi

# Download project data
./download.sh
