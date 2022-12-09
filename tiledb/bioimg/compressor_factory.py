from abc import ABC
from typing import Generic, TypeVar

import tiledb

T = TypeVar("T", tiledb.filter.ZstdFilter, tiledb.filter.WebpFilter)


class CompressorArguments(ABC, Generic[T]):
    pass


class WebpArguments(CompressorArguments[tiledb.filter.WebpFilter]):
    def __init__(
        self,
        quality: int,
        lossless: bool = False,
        image_format: tiledb.filter.lt.WebpInputFormat = tiledb.filter.lt.WebpInputFormat.WEBP_RGB,
    ):
        self._image_format = image_format
        self._quality = quality
        self._lossless = lossless

    @property
    def quality(self) -> int:
        return self._quality

    @property
    def lossless(self) -> bool:
        return self._lossless

    @property
    def image_format(self) -> tiledb.filter.lt.WebpInputFormat:
        return self._image_format

    @image_format.setter
    def image_format(self, f: tiledb.filter.lt.WebpInputFormat) -> None:
        self._image_format = f


class ZstdArguments(CompressorArguments[tiledb.filter.ZstdFilter]):
    def __init__(self, level: int):
        self._level = level

    @property
    def level(self) -> int:
        return self._level


def createCompressor(arguments: CompressorArguments[T]) -> T:
    if isinstance(arguments, WebpArguments):
        return tiledb.filter.WebpFilter(
            input_format=arguments.image_format,
            quality=arguments.quality,
            lossless=arguments.lossless,
        )
    elif isinstance(arguments, ZstdArguments):
        return tiledb.filter.ZstdFilter(level=arguments.level)
    else:
        raise Exception()
