from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import tiledb

T = TypeVar("T", bound=tiledb.filter.Filter)


class CompressorArguments(ABC, Generic[T]):
    pass


class WebpArguments(CompressorArguments[tiledb.filter.WebpFilter]):

    def __init__(self, quality: int, lossless: bool):
        self._quality = quality
        self._lossless = lossless

    @property
    def quality(self):
        return self._quality

    @property
    def lossless(self):
        return self._lossless


class ZstdArguments(CompressorArguments[tiledb.filter.ZstdFilter]):

    def __init__(self, level: int):
        self._level = level

    @property
    def level(self):
        return self._level


def createCompressor(arguments: CompressorArguments[T]) -> T:
    if isinstance(arguments, WebpArguments):
        return tiledb.filter.WebpFilter(input_format=tiledb.filter.lt.WebpInputFormat.WEBP_RGB, quality=arguments.quality, lossless=arguments.lossless)
    elif isinstance(arguments, ZstdArguments):
        return tiledb.filter.ZstdFilter(level=arguments.level)
    else:
        raise Exception()
