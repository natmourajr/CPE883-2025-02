#!/bin/bash

# Download CORA from UCSC and extract into a structured format
TARGET_DIR="../../../datasets/cora"
mkdir -p "$TARGET_DIR"

echo "Downloading CORA from UCSC..."
wget -q --show-progress https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz -P "$TARGET_DIR"

echo "Extracting..."
tar -xzf "$TARGET_DIR/cora.tgz" -C "$TARGET_DIR" --strip-components=1

echo "Cleaning up..."
find "$TARGET_DIR" -type f -exec sed -i 's/\t/ /g' {} \;  

rm "$TARGET_DIR/cora.tgz"

echo "CORA dataset ready in $TARGET_DIR"
echo "Files:"
ls -lh "$TARGET_DIR"

echo "done!"