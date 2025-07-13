#!/bin/bash

URL="https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
TARGET_DIR="../datasets/"
FILE_TO_EXTRACT="processed.cleveland.data" 

mkdir -p "$TARGET_DIR"
FILENAME=$(basename "$URL")
echo $FILENAME

echo "downloading $URL..."
curl -L "$URL" -o "$FILENAME"

if [ -f "$FILENAME" ]; then
    echo "Extracting $FILE_TO_EXTRACT into $TARGET_DIR/heart_disease"
    unzip "$FILENAME" "$FILE_TO_EXTRACT" -d "$TARGET_DIR/heart_disease"
    
    rm "$FILENAME"
    
    echo "Running prepare_heart.py..."
    python3 prepare_heart.py
    
    echo "Done."
else
    echo "Failed to download the file."
fi