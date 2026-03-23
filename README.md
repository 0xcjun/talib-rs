<p align="center">
  <h1 align="center">talib-rs</h1>
  <p align="center">
    Pure Rust Technical Analysis Library — Drop-in Replacement for TA-Lib
  </p>
  <p align="center">
    <a href="README.zh-CN.md">中文</a> · <a href="BENCHMARK.md">Benchmarks</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/indicators-155-blue" alt="155 indicators" />
  <img src="https://img.shields.io/badge/tests-353_accuracy-green" alt="353 accuracy tests" />
  <img src="https://img.shields.io/badge/unsafe-zero-brightgreen" alt="zero unsafe" />
  <img src="https://img.shields.io/badge/precision-bit--exact-brightgreen" alt="bit-exact" />
  <img src="https://img.shields.io/badge/C_deps-zero-orange" alt="zero C deps" />
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-lightgrey" alt="BSD-3-Clause" />
</p>

---

## Why talib-rs?

[TA-Lib](https://ta-lib.org/) has been the industry standard for technical analysis since 1999, but its C implementation brings well-known pain points: difficult compilation on Windows/macOS, fragile Python wrapper builds, and no modern language safety guarantees. **talib-rs** solves all of these while delivering identical results.

| | C TA-Lib | talib-rs |
|---|---|---|
| Language | C (1999) | Rust (2024) |
| Installation | Requires C compiler + system libs | `pip install talib-rs` |
| Accuracy | Reference implementation | **Bit-exact match** (diff=0, 353 accuracy tests) |
| Performance | Baseline | **1.29x avg faster** (zero unsafe, O(n) + iterator vectorization) |
| Memory safety | Manual management | Guaranteed by Rust |
| Python integration | Cython wrapper | PyO3 zero-copy |
| Indicators | 155 | 155 (100% coverage) |

### Key Technical Advantages

- **Bit-exact precision** — Not "approximately equal", but `diff=0` against C TA-Lib across 1M+ data points. Verified with 353 accuracy tests covering every function, every parameter combination, and extreme edge cases.

- **O(n) algorithms everywhere** — STDDEV, CORREL, BETA, LINEARREG, KAMA, TRIMA all use incremental sliding-window formulas. Period-independent performance: `CORREL(period=200)` runs in the same time as `CORREL(period=10)`.

- **SIMD acceleration** — `wide` crate f64x4 vectorization for reduction operations (sum, sum-of-squares). ~3x speedup on aggregate computations used by SMA, BBANDS, STDDEV, VAR.

- **Zero-copy NumPy bridge** — Input arrays read directly from NumPy memory via `PyReadonlyArray1::as_slice()`. Output arrays transferred via `PyArray1::from_vec()` with ownership move, no memcpy.

- **Monotonic deque optimization** — AROON, WILLR, MAX, MIN, MIDPOINT, MIDPRICE use O(n) sliding-window extrema instead of O(n×p) brute-force scan.

- **Zero unsafe** — Entire codebase uses safe Rust. No `unsafe` blocks, no `get_unchecked`. Iterator-based patterns enable LLVM auto-vectorization while maintaining full bounds safety.

## Installation

```bash
pip install talib-rs
```

No C compiler, no system libraries, no build failures. Pre-built wheels for Linux/macOS/Windows × Python 3.9–3.13.

## Quick Start

### Function API (TA-Lib compatible)

```python
import talib_rs as talib
import numpy as np

close = np.random.random(1000) * 100

# Trend
sma  = talib.SMA(close, timeperiod=20)
ema  = talib.EMA(close, timeperiod=12)
kama = talib.KAMA(close, timeperiod=30)
upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# Momentum
rsi  = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3)

# Volatility & Volume
atr = talib.ATR(high, low, close, timeperiod=14)
obv = talib.OBV(close, volume)

# Pattern Recognition (61 candlestick patterns)
engulfing = talib.CDLENGULFING(open, high, low, close)  # -100 / 0 / 100
hammer    = talib.CDLHAMMER(open, high, low, close)

# Metadata
print(len(talib.get_functions()))       # 155
print(talib.get_function_groups().keys())
# ['Overlap Studies', 'Momentum Indicators', 'Pattern Recognition', ...]
```

### Abstract API

```python
from talib_rs.abstract import Function

# Create function with introspection
rsi = Function('RSI')
print(rsi.info['group'])         # 'Momentum Indicators'
print(rsi.input_names)           # OrderedDict([('price', ['close'])])
print(rsi.parameters)            # OrderedDict([('timeperiod', 14)])
print(rsi.output_names)          # ['real']
print(rsi.lookback)              # 14

# Call with dict or DataFrame
data = {'close': close, 'high': high, 'low': low, 'open': open_, 'volume': volume}
result = rsi(data)

# Override parameters
rsi.parameters = {'timeperiod': 21}
result = rsi(data)

# One-liner imports
from talib_rs.abstract import SMA, MACD, BBANDS
sma_result = SMA(data, timeperiod=50)
```

### Stream API

```python
from talib_rs import stream

# Returns only the latest value (scalar), not the full array
latest_rsi  = stream.RSI(close, timeperiod=14)          # float
latest_macd = stream.MACD(close, 12, 26, 9)             # (float, float, float)
latest_sma  = stream.SMA(close, timeperiod=20)           # float
```

## Performance

### 1M Data Points (Apple M4, `--release`, LTO fat)

| Indicator | Rust (μs) | C TA-Lib (μs) | Speedup |
|-----------|-------:|--------:|--------:|
| MIDPOINT(14) | 4,001 | 39,460 | **9.86x** |
| BBANDS(20) | 1,164 | 3,499 | **3.01x** |
| TEMA(20) | 1,419 | 3,904 | **2.75x** |
| LINEARREG(14) | 1,846 | 3,974 | **2.15x** |
| TRIX(15) | 1,989 | 4,094 | **2.06x** |
| STOCH(5,3,3) | 3,756 | 6,749 | **1.80x** |
| SMA(20) | 585 | 909 | **1.56x** |
| MACD(12,26,9) | 2,834 | 4,262 | **1.50x** |
| ADX(14) | 3,719 | 6,065 | **1.63x** |
| RSI(14) | 3,734 | 3,812 | 1.02x |
| EMA(20) | 1,314 | 1,193 | 0.91x |
| ATR(14) | 4,072 | 3,961 | 0.97x |

**90 indicators benchmarked** | 40 faster | 24 equal | 26 slower | Average speedup: **1.29x** | Median: **1.02x**

### Algorithm Complexity Comparison

All optimizable indicators match C TA-Lib's O(n) complexity:

| Indicator | Before | After | Speedup @p=200 |
|-----------|--------|-------|----------------|
| STDDEV | O(n×p) | O(n) | 45x |
| CORREL | O(n×p) | O(n) | 127x |
| BETA | O(n×p) | O(n) | 174x |
| LINEARREG | O(n×p) | O(n) | 102x |
| KAMA | O(n×p) | O(n) | 48x |
| TRIMA | O(n×p) | O(n) | 48x |
| WMA | O(n×p) | O(n) | 4.7x |

Full benchmark: 90 indicators × 4 dataset sizes (1K/10K/100K/1M) → [BENCHMARK.md](BENCHMARK.md)

## Indicators (155)

### Overlap Studies (16)
SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA, BBANDS, SAR, SAREXT, MIDPOINT, MIDPRICE, MAVP, HT_TRENDLINE

### Momentum Indicators (30)
RSI, MACD, MACDEXT, MACDFIX, STOCH, STOCHF, STOCHRSI, ADX, ADXR, CCI, MOM, ROC, ROCP, ROCR, ROCR100, WILLR, APO, PPO, BOP, CMO, AROON, AROONOSC, MFI, TRIX, ULTOSC, DX, PLUS_DI, MINUS_DI, PLUS_DM, MINUS_DM

### Pattern Recognition (61)
CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR, CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, CDLINNECK, CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR, CDLSHORTLINE, CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP, CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS

### Volatility (3)
ATR, NATR, TRANGE

### Volume (3)
AD, ADOSC, OBV

### Price Transform (4)
AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

### Statistic Functions (9)
STDDEV, VAR, BETA, CORREL, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, TSF

### Math Transform (15)
ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH

### Math Operators (9)
ADD, SUB, MULT, DIV, MAX, MAXINDEX, MIN, MININDEX, SUM

### Cycle Indicators (5)
HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

## Testing & Verification

```
155/155 functions · 353 accuracy tests · 6 dataset types · bit-exact with C TA-Lib · 0 failures
```

### Test Matrix

| Suite | Cases | Description |
|-------|------:|------------|
| **Rust unit tests** | 55 | Core algorithm, SIMD, sliding window, edge cases |
| **Accuracy cross-validation** | 353 | 155 functions × 6 datasets, rtol=1e-10, 61 pattern exact match |

### Verification Methodology

1. **Numerical precision**: `max(|ta_rs[i] - c_talib[i]|) = 0` for all non-NaN positions. Not "close to zero" — exactly zero.
2. **NaN alignment**: Lookback NaN positions match exactly between talib-rs and C TA-Lib. Verified for first valid index and total NaN count.
3. **Accumulation stability**: EMA/RSI/MACD tested on 1,000,000 data points with zero accumulated drift.
4. **Pattern recognition**: All 61 candlestick patterns produce identical signals (-100/0/100) across 4 datasets × 5000 bars.

```bash
cargo test                                        # 55 Rust tests
pytest tests/test_exhaustive.py -v                # 155/155 cross-validation
pytest tests/test_full_coverage.py -q             # 620 consistency + edge
```

## Architecture

```
talib-rs/
├── crates/
│   ├── talib-rs-core/                    # Pure Rust library (no Python dependency)
│   │   ├── src/
│   │   │   ├── overlap/               # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA,
│   │   │   │                          # T3, MAMA, BBANDS, SAR, MIDPOINT, ...
│   │   │   ├── momentum/              # RSI, MACD, STOCH, ADX, CCI, MOM, ROC,
│   │   │   │                          # WILLR, AROON, MFI, TRIX, ULTOSC, DX, ...
│   │   │   ├── volatility/            # ATR, NATR, TRANGE
│   │   │   ├── volume/                # AD, ADOSC, OBV
│   │   │   ├── pattern/               # 61 candlestick patterns with real logic
│   │   │   ├── statistic/             # STDDEV, VAR, BETA, CORREL, LINEARREG, TSF
│   │   │   ├── cycle/                 # Hilbert Transform (ring buffer optimized)
│   │   │   ├── math_transform/        # sin, cos, sqrt, exp, ln, ... (loop-unrolled)
│   │   │   ├── math_operator/         # add, sub, max, min, sum (monotonic deque)
│   │   │   ├── price_transform/       # avgprice, medprice, typprice, wclprice
│   │   │   ├── simd.rs                # f64x4 SIMD: sum, sum_sq_diff (3x speedup)
│   │   │   ├── sliding_window.rs      # Monotonic deque: O(n) sliding max/min
│   │   │   ├── ma_type.rs             # MA type dispatcher (9 types)
│   │   │   └── error.rs               # TaError enum
│   │   └── benches/
│   │       └── simd_bench.rs          # Criterion benchmarks
│   │
│   └── talib-rs-python/                  # PyO3 bindings
│       └── src/
│           ├── func_api.rs            # 155 #[pyfunction] definitions
│           ├── metadata.rs            # get_functions(), get_function_groups()
│           └── conversion.rs          # Zero-copy NumPy ↔ Rust
│
├── python/talib_rs/
│   ├── __init__.py                    # import talib_rs as talib
│   ├── abstract.py                    # Function class with introspection
│   └── stream.py                      # Latest-value-only wrappers
│
├── tests/
│   ├── test_exhaustive.py             # 155/155 cross-validation
│   ├── test_full_coverage.py          # Consistency + edge (620 tests)
│   └── accuracy/                      # Focused accuracy tests
│
├── benches/
│   ├── generate_report.py             # → BENCHMARK.md
│   └── python_benches/                # pytest-benchmark suites
│
└── .github/workflows/
    ├── ci.yml                         # Test + lint (3 OS × 4 Python versions)
    └── release.yml                    # Build wheels + publish to PyPI
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SIMD crate | `wide` 0.7 | Stable Rust, no nightly required |
| Sliding extrema | Monotonic deque | Fused dual-extremum scan, amortized O(n) |
| Variance | E[X²]−E[X]² | Single-pass sliding window, no Welford two-pass |
| Hilbert Transform | Ring buffer | 10 Vec → fixed [f64; 64], 80% less heap allocation |
| EMA/DEMA/TEMA/T3 | In-place layered | No intermediate Vec from NaN filtering |
| WMA | Incremental recurrence | WS_new = WS_old − S_old + p×x_new, O(1) per step |
| Python module name | `talib_rs` | `import talib_rs as talib` for drop-in usage |
| Output format | `Vec<f64>` → `PyArray1` | Ownership transfer, near-zero-copy |
| Unsafe policy | Zero `unsafe` | Iterator patterns + safe indexing; LLVM auto-vectorizes zip chains |
| NaN convention | Fill lookback with NaN | Matches C TA-Lib exactly |

## Development

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python dependencies
pip install maturin numpy pytest pytest-benchmark
```

### Build & Test

```bash
maturin develop --release              # Build Python package
cargo test                             # 55 Rust tests
pytest tests/ -q                       # 353 accuracy tests
```

### Benchmarking

```bash
cargo bench --bench simd_bench         # Rust SIMD micro-benchmarks
pytest benches/ --benchmark-sort=name  # Python vs C TA-Lib
python benches/generate_report.py      # Generate BENCHMARK.md
```

### CI/CD

- **CI**: Rust test + clippy + fmt + Python build on Ubuntu/macOS/Windows × Python 3.10–3.13
- **Release**: Tag `v*` triggers manylinux/macOS/Windows wheel builds + PyPI publish

## Migration from C TA-Lib

```bash
# Remove C TA-Lib
pip uninstall TA-Lib

# Install talib-rs
pip install talib-rs

# Update imports: import talib_rs as talib
```

## License

BSD-3-Clause
