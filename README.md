# ta-rs

[English](#english) | [中文](#中文)

---

<a id="english"></a>

Pure Rust technical analysis library — **drop-in replacement** for [TA-Lib](https://github.com/TA-Lib/ta-lib).

## Why ta-rs?

- **155 indicators** — complete TA-Lib coverage, all in pure Rust
- **Bit-exact** — verified against C TA-Lib with 600+ automated tests, diff=0
- **Zero-copy** NumPy integration via PyO3 — no data copying between Python and Rust
- **O(n) algorithms** — STDDEV, CORREL, LINEARREG, KAMA, TRIMA use sliding window, not O(n×p)
- **SIMD accelerated** — `wide` f64x4 for sum/variance reductions (~3x speedup)
- **No C dependencies** — `pip install` just works, no compiler needed
- **API compatible** — `import talib` with identical function names, parameters, and return types

## Installation

```bash
pip install ta-rs
```

## Quick Start

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

# Abstract API (accepts dict / DataFrame)
from talib.abstract import SMA, RSI, Function
result = SMA({'close': close}, timeperiod=20)
func = Function('RSI')
func.parameters = {'timeperiod': 10}
result = func({'close': close})

# Stream API (latest value only)
from talib import stream
latest_rsi = stream.RSI(close, timeperiod=14)

# Discover
print(talib.get_functions())        # 155 functions
print(talib.get_function_groups())  # 10 categories
```

## Performance

Benchmarked on 100,000 data points (Apple M-series, `--release`):

| Indicator | ta-rs (μs) | C TA-Lib (μs) | Ratio |
|-----------|----------:|-------------:|------:|
| SMA | 62 | 65 | **1.05x** |
| EMA | 135 | 136 | 1.00x |
| RSI | 369 | 371 | 1.00x |
| MACD | 610 | 610 | 1.00x |
| BBANDS | 447 | 447 | 1.00x |
| ADX | 1996 | 1978 | 0.99x |
| ATR | 403 | 403 | 1.00x |
| OBV | 222 | 225 | 1.01x |

90 indicators benchmarked across 1K/10K/100K — see [BENCHMARK.md](BENCHMARK.md) for full results.

All indicators use the same O(n) algorithm complexity as C TA-Lib. Performance is identical within measurement noise.

## Indicators (155)

| Category | Count | Examples |
|----------|------:|---------|
| Overlap Studies | 16 | SMA, EMA, WMA, DEMA, TEMA, BBANDS, SAR, KAMA, T3, MAMA |
| Momentum | 30 | RSI, MACD, STOCH, ADX, CCI, MOM, ROC, WILLR, AROON, MFI |
| Pattern Recognition | 61 | CDLDOJI, CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, … |
| Math Transform | 15 | SIN, COS, SQRT, EXP, LN, LOG10, … |
| Math Operators | 9 | ADD, SUB, MULT, DIV, MAX, MIN, SUM |
| Statistic | 9 | STDDEV, VAR, BETA, CORREL, LINEARREG, TSF |
| Cycle | 5 | HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE |
| Price Transform | 4 | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE |
| Volatility | 3 | ATR, NATR, TRANGE |
| Volume | 3 | AD, ADOSC, OBV |

## Testing

```
155/155 functions covered | 600+ test cases | bit-exact with C TA-Lib
```

| Suite | Tests | What it verifies |
|-------|------:|-----------------|
| Rust unit tests | 50 | Algorithm correctness, SIMD, edge cases |
| Exhaustive cross-validation | 243 | Every function vs C TA-Lib, multiple params |
| Full consistency (multi-dataset) | 310 | 155 functions × 2 datasets, diff=0 |
| Edge cases | 310 | Constant input, 100K data, boundary conditions |
| Blind spot verification | 52 | Flat data, 1M accumulation, NaN inputs, patterns |

```bash
# Run all tests
cargo test                                        # Rust
pytest tests/test_full_coverage.py -q             # Consistency + Edge
pytest tests/test_exhaustive.py -q                # 155/155 cross-validation
```

## Architecture

```
crates/
├── ta-rs-core/          # Pure Rust indicator library
│   ├── src/overlap/     #   SMA, EMA, BBANDS, SAR, …
│   ├── src/momentum/    #   RSI, MACD, ADX, …
│   ├── src/pattern/     #   61 candlestick patterns
│   ├── src/simd.rs      #   SIMD f64x4 acceleration
│   └── src/sliding_window.rs  # Monotonic deque O(n) extrema
└── ta-rs-python/        # PyO3 bindings (zero-copy NumPy)

python/talib/
├── __init__.py          # Drop-in: import talib
├── abstract.py          # Function class, dict/DataFrame inputs
└── stream.py            # Latest-value-only computation
```

## Development

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin numpy pytest pytest-benchmark

maturin develop --release          # Build
cargo test                         # Rust tests
pytest tests/ -q                   # Python tests
cargo bench --bench simd_bench     # SIMD benchmarks
python benches/generate_report.py  # Full benchmark report
```

## License

MIT

---

<a id="中文"></a>

# ta-rs

纯 Rust 技术分析库 — [TA-Lib](https://github.com/TA-Lib/ta-lib) 的**完全替代品**。

## 为什么选择 ta-rs？

- **155 个指标** — 完整覆盖 TA-Lib，全部用纯 Rust 实现
- **位精度一致** — 与 C TA-Lib 通过 600+ 自动化测试验证，diff=0
- **零拷贝** NumPy 集成 — 通过 PyO3，Python 和 Rust 之间无数据复制
- **O(n) 算法** — STDDEV、CORREL、LINEARREG、KAMA、TRIMA 使用滑动窗口，非 O(n×p)
- **SIMD 加速** — `wide` f64x4 用于求和/方差归约（~3x 加速）
- **无 C 依赖** — `pip install` 即可使用，无需编译器
- **API 兼容** — `import talib` 直接替换，函数名、参数、返回值完全一致

## 安装

```bash
pip install ta-rs
```

## 快速开始

```python
import talib
import numpy as np

close = np.random.random(100)

# 移动平均线
sma = talib.SMA(close, timeperiod=20)
ema = talib.EMA(close, timeperiod=12)

# 动量指标
rsi = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# 波动率
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# 抽象 API (接受字典 / DataFrame)
from talib.abstract import SMA, RSI, Function
result = SMA({'close': close}, timeperiod=20)
func = Function('RSI')
func.parameters = {'timeperiod': 10}
result = func({'close': close})

# 流式 API (仅返回最新值)
from talib import stream
latest_rsi = stream.RSI(close, timeperiod=14)

# 查看所有函数
print(talib.get_functions())        # 155 个函数
print(talib.get_function_groups())  # 10 个分类
```

## 性能

100,000 条数据基准测试 (Apple M 系列, `--release`):

| 指标 | ta-rs (μs) | C TA-Lib (μs) | 比率 |
|------|----------:|-------------:|-----:|
| SMA | 62 | 65 | **1.05x** |
| EMA | 135 | 136 | 1.00x |
| RSI | 369 | 371 | 1.00x |
| MACD | 610 | 610 | 1.00x |
| BBANDS | 447 | 447 | 1.00x |
| ADX | 1996 | 1978 | 0.99x |
| ATR | 403 | 403 | 1.00x |
| OBV | 222 | 225 | 1.01x |

90 个指标在 1K/10K/100K 三种数据量下完整测试 — 完整结果见 [BENCHMARK.md](BENCHMARK.md)。

所有指标与 C TA-Lib 使用相同的 O(n) 算法复杂度，性能在测量误差范围内完全一致。

## 指标列表 (155 个)

| 分类 | 数量 | 示例 |
|------|-----:|------|
| 趋势叠加 | 16 | SMA, EMA, WMA, DEMA, TEMA, BBANDS, SAR, KAMA, T3, MAMA |
| 动量 | 30 | RSI, MACD, STOCH, ADX, CCI, MOM, ROC, WILLR, AROON, MFI |
| K线形态 | 61 | CDLDOJI, CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, … |
| 数学变换 | 15 | SIN, COS, SQRT, EXP, LN, LOG10, … |
| 数学运算 | 9 | ADD, SUB, MULT, DIV, MAX, MIN, SUM |
| 统计 | 9 | STDDEV, VAR, BETA, CORREL, LINEARREG, TSF |
| 周期 | 5 | HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE |
| 价格变换 | 4 | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE |
| 波动率 | 3 | ATR, NATR, TRANGE |
| 成交量 | 3 | AD, ADOSC, OBV |

## 测试覆盖

```
155/155 函数全覆盖 | 600+ 测试用例 | 与 C TA-Lib 位精度一致
```

| 测试套件 | 数量 | 验证内容 |
|---------|-----:|---------|
| Rust 单元测试 | 50 | 算法正确性、SIMD、边界条件 |
| 穷尽交叉验证 | 243 | 每个函数 vs C TA-Lib，多参数组合 |
| 全量一致性 (多数据集) | 310 | 155 函数 × 2 数据集，diff=0 |
| 边界测试 | 310 | 常数输入、100K 数据、极端条件 |
| 盲点验证 | 52 | 全平数据、1M 累积误差、NaN 输入、形态逐值对比 |

```bash
# 运行所有测试
cargo test                                        # Rust
pytest tests/test_full_coverage.py -q             # 一致性 + 边界
pytest tests/test_exhaustive.py -q                # 155/155 交叉验证
```

## 架构

```
crates/
├── ta-rs-core/          # 纯 Rust 指标库
│   ├── src/overlap/     #   SMA, EMA, BBANDS, SAR, …
│   ├── src/momentum/    #   RSI, MACD, ADX, …
│   ├── src/pattern/     #   61 种 K 线形态
│   ├── src/simd.rs      #   SIMD f64x4 加速
│   └── src/sliding_window.rs  # 单调队列 O(n) 极值
└── ta-rs-python/        # PyO3 绑定 (零拷贝 NumPy)

python/talib/
├── __init__.py          # 直接替换: import talib
├── abstract.py          # Function 类, 字典/DataFrame 输入
└── stream.py            # 仅返回最新值的流式计算
```

## 开发

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin numpy pytest pytest-benchmark

maturin develop --release          # 构建
cargo test                         # Rust 测试
pytest tests/ -q                   # Python 测试
cargo bench --bench simd_bench     # SIMD 基准
python benches/generate_report.py  # 生成完整性能报告
```

## 许可证

MIT
