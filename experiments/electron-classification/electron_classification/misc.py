from typing import Any, Generator, Iterable
from pathlib import Path


N_RINGS = 100


def iterable_to_generator(iterable: Iterable[Any]) -> Generator[Any, None, None]:
    """
    Converts an iterable to a generator.

    Parameters
    ----------
    iterable : Iterable[Any]
        The iterable to be converted into a generator.

    Returns
    -------
    Generator[Any, None, None]
        A generator that yields items from the input iterable.
    """
    yield from iterable


def list_by_pattern(files: Iterable[Path], pattern: str) -> Generator[Path, None, None]:
    """
    Get files from an iterable of paths that match a specific pattern.

    Parameters
    ----------
    files : Iterable[Path]
        An iterable of Path objects to search for matching files.
    pattern : str
        The pattern to match (e.g., '*.parquet').

    Returns
    -------
    Generator[Path, None, None]
        A generator that yields paths of matching files.
    """
    for file in files:
        if file.is_dir():
            yield from list_by_pattern(file.iterdir(), pattern)
        else:
            if file.match(pattern):
                yield file
