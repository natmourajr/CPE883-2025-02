
#!/bin/bash

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TARGET_DIR="../datasets/"

mkdir -p "$TARGET_DIR"
FILENAME=$(basename "$URL")

echo "downloading $URL..."
curl -L "$URL" -o "$FILENAME"

echo "extracting $FILENAME into $TARGET_DIR"
tar -xzf "$FILENAME" -C "$TARGET_DIR"

rm "$FILENAME"

echo "Running prepare_data.py..."
python3 prepare_cifar.py

echo "Done."
