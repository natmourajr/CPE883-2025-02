import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import random
import json

# parse args for each dataset

PKL_DIR = Path(
    '../datasets/cifar-10-batches-py/')
IMG_OUT_DIR = PKL_DIR / 'images'
ANNOTATION_FILE = PKL_DIR / 'annotations.txt'
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 50
TRAIN_RATIO = 0.9


def unflatten_image(flat_row):
    """Convert a 3072-length flat array to a 32x32x3 RGB image."""
    r = flat_row[0:1024].reshape((32, 32))
    g = flat_row[1024:2048].reshape((32, 32))
    b = flat_row[2048:].reshape((32, 32))
    return np.stack([r, g, b], axis=-1)  # (32, 32, 3)


def save_set(name, dataset):
    ann_file = PKL_DIR / f"{name}_annotations.txt"
    with open(ann_file, 'w') as ann_f:
        for fname, img_array, label in dataset:
            Image.fromarray(img_array).save(IMG_OUT_DIR / fname, format='JPEG')
            ann_f.write(f"{fname} {label}\n")
    print(f"{name.capitalize()} set: {len(dataset)} samples saved.")


def main():
    train_data = []

    # unpack train set
    for pkl_file in PKL_DIR.glob('data_batch_*'):
        print(f'Processing: {pkl_file}')
        with open(pkl_file, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')

        data = batch['data']
        labels = batch['labels']

        for i, flat_img in enumerate(data):
            img_array = unflatten_image(flat_img)
            filename = f"{pkl_file.stem}_{i:05}.jpg"
            train_data.append((filename, img_array, labels[i]))

        pkl_file.unlink()

    # unpack test set
    test_data = []
    test_file = PKL_DIR / 'test_batch'
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')

    data = batch['data']
    labels = batch['labels']
    for i, flat_img in enumerate(data):
        img_array = unflatten_image(flat_img)
        filename = f"{test_file.stem}_{i:05}.jpg"
        test_data.append((filename, img_array, labels[i]))

    test_file.unlink()

    save_set("train", train_data)
    save_set("test", test_data)

    # unpack meta file (class info)
    meta_file = PKL_DIR / 'batches.meta'

    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')  # Important: latin1

    label_names = meta['label_names']
    label_dict = {str(i): name for i, name in enumerate(label_names)}
    label_json_file = PKL_DIR / 'label_names.json'
    with open(label_json_file, 'w') as jf:
        json.dump(label_dict, jf, indent=2)

    meta_file.unlink()

    print(f"✅ Saved label names to {label_json_file}")

    print("images + labels extracted")


if __name__ == '__main__':
    random.seed(SEED)
    main()
