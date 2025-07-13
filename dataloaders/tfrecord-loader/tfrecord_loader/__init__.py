# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=pt-br#tftrainexample

from pathlib import Path
from typing import Any, Dict, Generator, Callable

import tensorflow as tf


def bytes_feature(value: str | bytes) -> tf.train.Feature:
    """
    Returns a bytes_list from a string / byte.

    Parameters
    ----------
    value : str | bytes
        The value to be converted into a bytes feature.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the bytes representation of the input value.
    """
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value: float) -> tf.train.Feature:
    """
    Returns a float_list from a float / double.

    Parameters
    ----------
    value : float
        The value to be converted into a float feature.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the float representation of the input value.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value: bool | int) -> tf.train.Feature:
    """
    Returns an int64_list from a bool / enum / int / long.

    Parameters
    ----------
    value : bool | int
        The value to be converted into an int64 feature.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the int64 representation of the input value.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tensor_feature(value: tf.Tensor) -> tf.train.Feature:
    """
    Returns a bytes_list from a Tensor.

    Parameters
    ----------
    value : tf.Tensor
        The Tensor to be converted into a bytes feature.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the serialized tensor.
    """
    return bytes_feature(tf.io.serialize_tensor(value))


TFRecordSchema = Dict[str, Callable[[Any], tf.train.Feature]]
TFRecordGenerator = Generator[Dict[str, Any], None, None]


def generator_to_tfrecord(
    generator: TFRecordGenerator,
    output_file: str | Path,
    schema: TFRecordSchema,
) -> str:
    # Write the `tf.train.Example` observations to the file.
    if isinstance(output_file, Path):
        output_file = str(output_file)
    with tf.io.TFRecordWriter(output_file) as writer:
        for sample in generator:
            feature = {
                key: feature_fn(sample[key]) for key, feature_fn in schema.items()
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            serialized_example = example.SerializeToString()
            writer.write(serialized_example)
    return output_file


N_RINGS = 100

RINGS_SCHEMA = {
    'id': tf.io.FixedLenFeature([], tf.int64),
    'et': tf.io.FixedLenFeature([], tf.float32),
    'eta': tf.io.FixedLenFeature([], tf.float32),
    'avgmu': tf.io.FixedLenFeature([], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.int64),
}
for i in range(N_RINGS):
    RINGS_SCHEMA[f'ring_{i}'] = tf.io.FixedLenFeature([], tf.float32)
