#!/bin/bash

JSON_FILE="AriaGen2PilotDataset_download_urls.json"
SEQUENCE="walk_0"
DIR="./datasets/$SEQUENCE/"

if [ ! -d "$DIR" ]; then
  echo "Directory $DIR does not exist. Creating it now..."
  mkdir -p "$DIR"
else
  echo "Directory $DIR already exists."
fi

jq -r '
  .sequences.'"$SEQUENCE"'
  | to_entries[]
  | [.value.filename, .value.download_url]
  | @tsv
' "$JSON_FILE" |
while IFS=$'\t' read -r filename url; do
    echo "Downloading $filename"
    curl -L --fail --retry 3 \
         -o "$DIR/$filename" \
         "$url"
done

cd datasets
cd $SEQUENCE
for z in *.zip; do
    echo "Extracting $z"
    unzip -o "$z"
done