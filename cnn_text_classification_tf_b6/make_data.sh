#! /usr/bin/env bash

# Exiting when any command fails
set -e

# Parameters
BUG_ID="cnn_text_classification_tf_b6"
DATA_DIRECTORY_PATH=~/annotest_subjects_data

# File id on Google drive
FILE_ID=1ryFZLV2A_LIjFuPrAwZnD3m6v2tntpAI

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
