from utils.data_faker import DataFaker
from argparse import ArgumentParser
from pathlib import Path

from utils import tfrecord

N_RINGS = 100


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Generate fake data based on a schema.")
    parser.add_argument('--class-ratio', type=float, required=True,
                        help="Ratio of samples in the class.")
    parser.add_argument('--n', type=int, required=True,
                        help="Number of samples to generate.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument('--output', type=str, required=True,
                        help="Output file name for the generated data.")
    args = parser.parse_args()
    args.output = Path(args.output)
    if args.output.suffix != '.tfrecord':
        raise ValueError("Only .tfrecord files are supported for output.")
    return args


def main(
    class_ratio: float,
    n: int,
    output_file: Path,
    seed: int | None = None
) -> None:
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

    # Generate the data
    # df = data_faker.generate(n)
    # df.to_parquet(output_file, index=False)

    tfrecord.generator_to_tfrecord(
        generator=data_faker.generator(n),
        output_file=output_file,
        schema=data_faker.tfrecord_schema
    )


if __name__ == '__main__':
    args = parse_args()
    main(
        class_ratio=args.class_ratio,
        n=args.n,
        output_file=args.output,
        seed=args.seed
    )
    print(f"Generated {args.n} samples with class ratio {args.class_ratio} and saved to {args.output}.")
