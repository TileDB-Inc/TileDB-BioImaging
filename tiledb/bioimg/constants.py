from typing import Literal

SpaceUnit = Literal[
    'angstrom', 'attometer', 'centimeter', 'decimeter', 'exameter', 'femtometer', 'foot', 'gigameter', 'hectometer', 'inch', 'kilometer', 'megameter', 'meter', 'micrometer', 'mile', 'millimeter', 'nanometer', 'parsec', 'petameter', 'picometer', 'terameter', 'yard', 'yoctometer', 'yottameter', 'zeptometer', 'zettameter']
TimeUnit = Literal[
    'attosecond', 'centisecond', 'day', 'decisecond', 'exasecond', 'femtosecond', 'gigasecond', 'hectosecond', 'hour', 'kilosecond', 'megasecond', 'microsecond', 'millisecond', 'minute', 'nanosecond', 'petasecond', 'picosecond', 'second', 'terasecond', 'yoctosecond', 'yottasecond', 'zeptosecond', 'zettasecond']

spaceUnitSymbolMap = {
    "Å": 'angstrom',
    "am": 'attometer',
    "cm": 'centimeter',
    "dm": 'decimeter',
    "Em": 'exameter',
    "fm": 'femtometer',
    "ft": 'foot',
    "Gm": 'gigameter',
    "hm": 'hectometer',
    "in": 'inch',
    "km": 'kilometer',
    "Mm": 'megameter',
    "m": 'meter',
    "µm": 'micrometer',
    "mi.": 'mile',
    "mm": 'millimeter',
    "nm": 'nanometer',
    "pc": 'parsec',
    "Pm": 'petameter',
    "pm": 'picometer',
    "Tm": 'terameter',
    "yd": 'yard',
    "ym": 'yoctometer',
    "Ym": 'yottameter',
    "zm": 'zeptometer',
    "Zm": 'zettameter'
}

timeUnitSymbolMap = {
    "as": 'attosecond',
    "cs": 'centisecond',
    "d": 'day',
    "ds": 'decisecond',
    "Es": 'exasecond',
    "fs": 'femtosecond',
    "Gs": 'gigasecond',
    "hs": 'hectosecond',
    "h": 'hour',
    "ks": 'kilosecond',
    "Ms": 'megasecond',
    "µs": 'microsecond',
    "ms": 'millisecond',
    "min": 'minute',
    "ns": 'nanosecond',
    "Ps": 'petasecond',
    "ps": 'picosecond',
    "s": 'second',
    "Ts": 'terasecond',
    "ys": 'yoctosecond',
    "Ys": 'yottasecond',
    "zs": 'zeptosecond',
    "Zs": 'zettasecond'
}