from typing import Any, List, Tuple, Dict
import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typer import Typer
from .misc import N_RINGS

# from rings_loader import (
#     float_feature,
#     int64_feature,
#     TFRecordSchema,
#     TFRecordGenerator
# )

SchemaDict = Dict[str, Tuple[str, Dict[str, Any]]]


class DataFaker:

    def __init__(self,
                 schema: SchemaDict = {},
                 seed: int | None = None):
        self.schema = schema
        self.numpy_generator = np.random.default_rng(seed)
        self.generators = {
            'constant': {
                'generator': self.constant,
                # 'tf_feature': float_feature
            },
            'gaussian': {
                'generator': self.normal,
                # 'tf_feature': float_feature
            },
            'normal': {
                'generator': self.normal,
                # 'tf_feature': float_feature
            },
            'uniform': {
                'generator': self.uniform,
                # 'tf_feature': float_feature
            },
            # Choice doesn't work properly with bytes_feature
            # should implement type checks to select the right feature type
            'choice': {
                'generator': self.choice,
                # 'tf_feature': int64_feature
            },
            'sequential': {
                'generator': self.sequential,
                # 'tf_feature': int64_feature
            }
        }

    # @cached_property
    # def tfrecord_schema(self) -> TFRecordSchema:
    #     """
    #     Returns the schema for TFRecord serialization.
    #     """
    #     return {
    #         column: self.generators[generator_name]['tf_feature']
    #         for column, (generator_name, _) in self.schema.items()
    #     }

    def add_field(self, column: str, generator_name: str, **params) -> None:
        if generator_name not in self.generators:
            raise ValueError(f"Generator '{generator_name}' is not supported.")
        self.schema[column] = (generator_name, params)

    def remove_field(self, column: str) -> None:
        if column in self.schema:
            self.schema.pop(column)
        else:
            raise KeyError(f"Column '{column}' does not exist in the schema.")

    def generate_df(self, n: int) -> pd.DataFrame:
        generated_data = {
            column: self.generators[generator_name]['generator'](**params, n=n)
            for column, (generator_name, params) in self.schema.items()
        }
        return pd.DataFrame.from_dict(generated_data)

    # def generator(self, n: int) -> TFRecordGenerator:
    #     for _ in range(n):
    #         yield {
    #             column: self.generators[generator_name]['generator'](
    #                 **params, n=1)[0]
    #             for column, (generator_name, params) in self.schema.items()
    #         }

    def constant(self, value: Any, n: int, dtype: npt.DTypeLike = np.float32) -> npt.NDArray[np.object_]:
        """Generate an array with a constant value."""
        return np.full(n, value, dtype=dtype)

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


app = Typer(
    name='faker',
    help='Utility for generating fake data.'
)


@app.command()
def rings_classification(
    n: int,
    class_ratio: float,
    output_dir: Path,
    seed: int | None = None
) -> None:

    data_faker = DataFaker(seed=seed)
    data_faker.add_field('id',
                         'sequential',
                         start=0)
    data_faker.add_field('label',
                         'constant',
                         value=0)
    data_faker.add_field('et',
                         'normal',
                         mean=0, std=1)
    data_faker.add_field('eta',
                         'normal',
                         mean=0, std=1)
    data_faker.add_field('avgmu',
                         'normal',
                         mean=0, std=1)
    for iring in range(N_RINGS):
        data_faker.add_field(f'ring_{iring}',
                             'normal',
                             mean=0, std=1)

    print(f"Generating {n} samples with class ratio {class_ratio}...")
    print('Generating class 0...')
    df = data_faker.generate_df(int(np.ceil((1-class_ratio) * n)))
    df.to_parquet(output_dir / 'class_0.parquet')
    # output_file = output_dir / 'class_0.tfrecord'
    # generator_to_tfrecord(
    #     generator=data_faker.generator(int(np.floor((1-class_ratio)*n))),
    #     output_file=output_file,
    #     schema=data_faker.tfrecord_schema
    # )

    data_faker.remove_field('label')
    data_faker.add_field('label',
                         'constant',
                         value=1)
    for iring in range(N_RINGS):
        data_faker.remove_field(f'ring_{iring}')
        data_faker.add_field(f'ring_{iring}',
                             'normal',
                             mean=10, std=1)
    print('Generating class 1...')
    df = data_faker.generate_df(int(np.ceil(class_ratio * n)))
    df.to_parquet(output_dir / 'class_1.parquet')
    # output_file = output_dir / 'class_1.tfrecord'
    # generator_to_tfrecord(
    #     generator=data_faker.generator(int(np.floor((1-class_ratio)*n))),
    #     output_file=output_file,
    #     schema=data_faker.tfrecord_schema
    # )
