from typer import Typer
from pathlib import Path

app = Typer()


@app.command()
def generate_fake_rings(
    n: int,
    class_ratio: float,
    output_file: Path,
    seed: int | None = None
) -> None:

    from tfrecord_loader import (
        generator_to_tfrecord,
        N_RINGS
    )
    from electron_classification.data_faker import DataFaker
    data_faker = DataFaker(seed=seed)
    data_faker.add_field('id',
                         'sequential',
                         start=0)
    data_faker.add_field('label',
                         'choice',
                         choices=[0, 1],
                         p=[1-class_ratio, class_ratio],
                         replace=True)
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

    generator_to_tfrecord(
        generator=data_faker.generator(n),
        output_file=output_file,
        schema=data_faker.tfrecord_schema
    )


if __name__ == "__main__":
    app()
