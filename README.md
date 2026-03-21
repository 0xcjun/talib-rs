# ta-rs

Pure Rust technical analysis library — **drop-in replacement** for [TA-Lib](https://github.com/TA-Lib/ta-lib).

## Features

- **155 indicators** implemented in pure Rust — complete TA-Lib coverage
- **Zero-copy** NumPy integration via PyO3
- **API compatible** with `import talib` — same function names, same parameters, same return types
- **Abstract API** — `from talib.abstract import SMA` with dict/DataFrame inputs
- **Stream API** — `from talib import stream` for latest-value-only computation
- **No C dependencies** — easy installation, no compilation headaches
- **Cross-platform** — works on macOS, Linux, and Windows

## Installation

```bash
pip install ta-rs
```

## Usage

```python
import talib
import numpy as np

close = np.random.random(100)

# Moving Averages
sma = talib.SMA(close, timeperiod=20)
ema = talib.EMA(close, timeperiod=12)

# Momentum
rsi = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# Volatility
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# Abstract API (dict / DataFrame input)
from talib.abstract import SMA, RSI
result = SMA({'close': close}, timeperiod=20)

# Stream API (latest value only)
from talib import stream
latest_rsi = stream.RSI(close, timeperiod=14)

# List all functions
print(talib.get_functions())        # 155 functions
print(talib.get_function_groups())  # 10 categories
```

## Performance

Benchmarked on 10,000 data points (Apple M-series, release build):

| Indicator | ta-rs (μs) | C TA-Lib (μs) | Speedup |
|-----------|-----------|--------------|---------|
| SMA | 6.4 | 6.7 | **1.05x** |
| EMA | 13.9 | 14.0 | 1.0x |
| RSI | 37.9 | 37.9 | 1.0x |
| MACD | 62.8 | 63.4 | 1.0x |
| BBANDS | 69.0 | 69.6 | 1.0x |
| ATR | 42.2 | 42.6 | 1.0x |
| ADX | 171.3 | 172.4 | 1.0x |
| OBV | 24.5 | 25.8 | **1.05x** |

Performance matches or exceeds the original C implementation, with the advantage of zero C dependencies and easy installation.

## Implemented Indicators (155 total)

| Category | Count | Examples |
|----------|-------|---------|
| Overlap Studies | 16 | SMA, EMA, WMA, DEMA, TEMA, BBANDS, SAR, KAMA, T3, MAMA |
| Momentum | 30 | RSI, MACD, STOCH, ADX, CCI, MOM, ROC, WILLR, AROON, MFI |
| Pattern Recognition | 61 | CDLDOJI, CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, etc. |
| Math Transform | 15 | SIN, COS, SQRT, EXP, LN, LOG10, etc. |
| Math Operators | 9 | ADD, SUB, MULT, DIV, MAX, MIN, SUM |
| Statistic | 9 | STDDEV, VAR, BETA, CORREL, LINEARREG, TSF |
| Cycle | 5 | HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE |
| Price Transform | 4 | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE |
| Volatility | 3 | ATR, NATR, TRANGE |
| Volume | 3 | AD, ADOSC, OBV |

## Development

```bash
# Install Rust + maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin numpy

# Build and install in development mode
maturin develop --release

# Run Rust tests (38 tests)
cargo test

# Run Python accuracy tests (19 tests, including vs C TA-Lib)
pytest tests/accuracy/ -v

# Run performance benchmarks
pytest benches/python_benches/ -v
```

## Architecture

```
crates/
├── ta-rs-core/     # Pure Rust indicator library (zero dependencies beyond std)
└── ta-rs-python/   # PyO3 Python bindings (zero-copy NumPy integration)

python/talib/
├── __init__.py     # Drop-in replacement: import talib
├── abstract.py     # Abstract API: Function class, dict/DataFrame inputs
└── stream.py       # Stream API: latest-value-only computation
```

## License

MIT
