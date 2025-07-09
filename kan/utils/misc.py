from typing import Any, Generator, Iterable


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
