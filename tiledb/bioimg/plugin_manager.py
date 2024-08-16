import importlib.metadata
from typing import Any, Mapping, Type

from tiledb.bioimg.converters import ImageConverterMixin, ImageReader, ImageWriter


class PluginManager:

    @staticmethod
    def load_readers() -> Mapping[str, ImageReader]:
        readers = {}
        for entry_point in importlib.metadata.entry_points()["bioimg.readers"]:
            readers[entry_point.name] = entry_point.load()
        return readers

    @staticmethod
    def load_writers() -> Mapping[str, ImageWriter]:
        writers = {}
        for entry_point in importlib.metadata.entry_points()["bioimg.writers"]:
            writers[entry_point.name] = entry_point.load()
        return writers

    @staticmethod
    def load_converters() -> Mapping[str, Type[ImageConverterMixin[Any, Any]]]:
        converters = {}
        for entry_point in importlib.metadata.entry_points()["bioimg.converters"]:
            converters[entry_point.name] = entry_point.load()
        return converters
