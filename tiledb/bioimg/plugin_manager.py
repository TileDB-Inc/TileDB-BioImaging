import sys
from typing import Any, Mapping, Type

from .converters.base import ImageConverterMixin, ImageReader, ImageWriter


def _load_entrypoints(name: str) -> Mapping[str, Any]:
    if sys.version_info < (3, 12):
        import importlib

        eps = importlib.metadata.entry_points()[name]
    else:
        from importlib.metadata import entry_points

        eps = entry_points(group=name)
    return {ep.name: ep.load() for ep in eps}


def load_readers() -> Mapping[str, ImageReader]:
    return _load_entrypoints("bioimg.readers")


def load_writers() -> Mapping[str, ImageWriter]:
    return _load_entrypoints("bioimg.writers")


def load_converters() -> Mapping[str, Type[ImageConverterMixin[Any, Any]]]:
    return _load_entrypoints("bioimg.converters")
