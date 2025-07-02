from functools import cached_property
from typing import Any, List, Tuple, Dict
import pandas as pd
import numpy as np
import numpy.typing as npt

from . import tfrecord

SchemaDict = Dict[str, Tuple[str, Dict[str, Any]]]


class DataFaker:

    def __init__(self,
                 schema: SchemaDict = {},
                 seed: int | None = None):
        self.schema = schema
        self.numpy_generator = np.random.default_rng(seed)
        self.generators = {
            'gaussian': {
                'generator': self.normal,
                'tf_feature': tfrecord.float_feature
            },
            'normal': {
                'generator': self.normal,
                'tf_feature': tfrecord.float_feature
            },
            'uniform': {
                'generator': self.uniform,
                'tf_feature': tfrecord.float_feature
            },
            # Choice doesn work properly with tfrecord.bytes_feature
            # should implement type checks to select the right feature type
            'choice': {
                'generator': self.choice,
                'tf_feature': tfrecord.int64_feature
            },
            'sequential': {
                'generator': self.sequential,
                'tf_feature': tfrecord.int64_feature
            }
        }

    @cached_property
    def tfrecord_schema(self) -> tfrecord.TFRecordSchema:
        """
        Returns the schema for TFRecord serialization.
        """
        return {
            column: self.generators[generator_name]['tf_feature']
            for column, (generator_name, _) in self.schema.items()
        }

    def add_field(self, column: str, generator_name: str, **params) -> None:
        if generator_name not in self.generators:
            raise ValueError(f"Generator '{generator_name}' is not supported.")
        self.schema[column] = (generator_name, params)

    def generate(self, n: int) -> pd.DataFrame:
        generated_data = {
            column: self.generators[generator_name](**params, n=n)
            for column, (generator_name, params) in self.schema.items()
        }
        return pd.DataFrame.from_dict(generated_data)

    def generator(self, n: int) -> tfrecord.TFRecordGenerator:
        for _ in range(n):
            yield {
                column: self.generators[generator_name]['generator'](
                    **params, n=1)[0]
                for column, (generator_name, params) in self.schema.items()
            }

    def gaussian(self, mean: float, std: float, n: int, dtype: npt.DTypeLike = np.float32) -> npt.NDArray[np.float32]:
        raise self.normal(mean, std, n, dtype)

    def normal(self, mean: float, std: float, n: int, dtype: npt.DTypeLike = np.float32) -> npt.NDArray[np.float32]:
        return self.numpy_generator.normal(loc=mean, scale=std, size=n).astype(dtype)

    def uniform(self, low: float, high: float, n: int, dtype: npt.DTypeLike = np.float32) -> npt.NDArray[np.float32]:
        return self.numpy_generator.uniform(low=low, high=high, size=n,).astype(dtype)

    def choice(self, choices: List[Any], n: int,
               replace: bool = True,
               p: List[float] | None = None) -> npt.NDArray[np.object_]:
        return self.numpy_generator.choice(
            choices,
            size=n,
            replace=replace,
            p=p,
            shuffle=False
        )

    def sequential(self, n: int, start: int = 0) -> npt.NDArray[np.int_]:
        """Generate a sequential array starting from `start` with a given `step`."""
        return np.arange(start, start+n, dtype=np.int64)
