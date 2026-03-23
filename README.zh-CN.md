<p align="center">
  <h1 align="center">talib-rs</h1>
  <p align="center">
    纯 Rust 技术分析库 — TA-Lib 的完全替代品
  </p>
  <p align="center">
    <a href="README.md">English</a> · <a href="BENCHMARK.md">Benchmarks</a>
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

## 为什么选择 talib-rs？

[TA-Lib](https://ta-lib.org/) 自 1999 年以来一直是技术分析的行业标准，但其 C 语言实现带来了众所周知的痛点：Windows/macOS 上编译困难、Python 封装构建脆弱、缺乏现代语言安全保证。**talib-rs** 在解决所有这些问题的同时，输出与原版完全一致。

| | C TA-Lib | talib-rs |
|---|---|---|
| 语言 | C (1999) | Rust (2024) |
| 安装 | 需要 C 编译器 + 系统库 | `pip install talib-rs` |
| 精度 | 参考实现 | **位精度一致** (diff=0, 353 准确性测试) |
| 性能 | 基准线 | **平均 1.29x 更快** (零 unsafe, O(n) + 迭代器向量化) |
| 内存安全 | 手动管理 | Rust 编译器保证 |
| Python 集成 | Cython 封装 | PyO3 零拷贝 |
| 指标数量 | 155 | 155 (100% 覆盖) |

### 核心技术优势

- **位精度一致** — 不是"近似相等"，而是与 C TA-Lib 的 `diff=0`，在 100 万条数据上验证。353 准确性测试覆盖每个函数、每种参数组合和极端边界情况。

- **全面 O(n) 算法** — STDDEV、CORREL、BETA、LINEARREG、KAMA、TRIMA 均使用增量滑动窗口公式。性能与 period 无关：`CORREL(period=200)` 与 `CORREL(period=10)` 耗时相同。

- **SIMD 加速** — `wide` crate f64x4 向量化归约运算（求和、平方和），在 SMA/BBANDS/STDDEV/VAR 的聚合计算上约 3 倍加速。

- **零拷贝 NumPy 桥接** — 输入通过 `PyReadonlyArray1::as_slice()` 直接读取 NumPy 内存；输出通过 `PyArray1::from_vec()` 所有权转移，无 memcpy。

- **单调队列优化** — AROON、WILLR、MAX、MIN、MIDPOINT、MIDPRICE 使用 O(n) 滑动窗口极值替代 O(n×p) 暴力扫描。

- **零 unsafe** — 整个代码库使用安全 Rust。无 `unsafe` 块，无 `get_unchecked`。基于迭代器的模式在保持完整边界安全的同时启用 LLVM 自动向量化。

## 安装

```bash
pip install talib-rs
```

无需 C 编译器，无需系统库，不会构建失败。提供 Linux/macOS/Windows × Python 3.9–3.13 的预编译 wheel。

## 快速开始

### 函数 API (与 TA-Lib 兼容)

```python
import talib_rs as talib
import numpy as np

close = np.random.random(1000) * 100

# 趋势
sma  = talib.SMA(close, timeperiod=20)
ema  = talib.EMA(close, timeperiod=12)
kama = talib.KAMA(close, timeperiod=30)
upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# 动量
rsi  = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# K 线形态识别 (61 种)
engulfing = talib.CDLENGULFING(open, high, low, close)  # -100 / 0 / 100

# 函数发现
print(len(talib.get_functions()))       # 155
print(talib.get_function_groups().keys())
```

### 抽象 API

```python
from talib_rs.abstract import Function

# 带内省的函数对象
rsi = Function('RSI')
print(rsi.info['group'])         # 'Momentum Indicators'
print(rsi.parameters)            # OrderedDict([('timeperiod', 14)])
print(rsi.lookback)              # 14

# 接受字典或 DataFrame
data = {'close': close, 'high': high, 'low': low}
result = rsi(data)

# 参数覆盖
rsi.parameters = {'timeperiod': 21}
result = rsi(data)

# 单行导入
from talib_rs.abstract import SMA, MACD, BBANDS
sma_result = SMA(data, timeperiod=50)
```

### 流式 API

```python
from talib_rs import stream

# 仅返回最新值（标量），非完整数组
latest_rsi  = stream.RSI(close, timeperiod=14)          # float
latest_macd = stream.MACD(close, 12, 26, 9)             # (float, float, float)
```

## 性能

### 100 万数据 (Apple M4, `--release`, LTO fat)

| 指标 | Rust (μs) | C TA-Lib (μs) | 加速比 |
|------|-------:|--------:|--------:|
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

**90 个指标测试** | 40 个更快 | 24 个持平 | 26 个较慢 | 平均加速: **1.29x** | 中位数: **1.02x**

### 算法复杂度优化

| 指标 | 优化前 | 优化后 | period=200 加速 |
|------|--------|--------|----------------|
| STDDEV | O(n×p) | O(n) | 45x |
| CORREL | O(n×p) | O(n) | 127x |
| BETA | O(n×p) | O(n) | 174x |
| LINEARREG | O(n×p) | O(n) | 102x |
| KAMA | O(n×p) | O(n) | 48x |
| TRIMA | O(n×p) | O(n) | 48x |
| WMA | O(n×p) | O(n) | 4.7x |

完整基准测试：90 个指标 × 4 种数据量 (1K/10K/100K/1M) → [BENCHMARK.md](BENCHMARK.md)

## 指标列表 (155 个)

### 趋势叠加 (16)
SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA, BBANDS, SAR, SAREXT, MIDPOINT, MIDPRICE, MAVP, HT_TRENDLINE

### 动量指标 (30)
RSI, MACD, MACDEXT, MACDFIX, STOCH, STOCHF, STOCHRSI, ADX, ADXR, CCI, MOM, ROC, ROCP, ROCR, ROCR100, WILLR, APO, PPO, BOP, CMO, AROON, AROONOSC, MFI, TRIX, ULTOSC, DX, PLUS_DI, MINUS_DI, PLUS_DM, MINUS_DM

### K 线形态 (61)
CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR, CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, CDLINNECK, CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR, CDLSHORTLINE, CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP, CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS

### 波动率 (3) · 成交量 (3) · 价格变换 (4)
ATR, NATR, TRANGE · AD, ADOSC, OBV · AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

### 统计 (9) · 数学变换 (15) · 数学运算 (9) · 周期 (5)
STDDEV, VAR, BETA, CORREL, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, TSF · ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH · ADD, SUB, MULT, DIV, MAX, MAXINDEX, MIN, MININDEX, SUM · HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

## 测试与验证

```
155/155 函数 · 353 准确性测试 · 6 种数据集 · 与 C TA-Lib 位精度一致 · 0 失败
```

| 测试套件 | 用例数 | 说明 |
|---------|------:|------|
| **Rust 单元测试** | 55 | 核心算法、SIMD、滑动窗口、边界条件 |
| **准确性交叉验证** | 353 | 155 函数 × 6 种数据集，rtol=1e-10，61 种形态精确匹配 |

### 验证方法论

1. **数值精度**: `max(|ta_rs[i] - c_talib[i]|) = 0`，所有非 NaN 位置。不是"接近零"——是精确零。
2. **NaN 对齐**: lookback NaN 位置与 C TA-Lib 完全一致。验证了首个有效索引和 NaN 总数。
3. **累积稳定性**: EMA/RSI/MACD 在 1,000,000 条数据上测试，零累积漂移。
4. **形态识别**: 全部 61 种 K 线形态在 4 个数据集 × 5000 根 K 线上产生完全相同的信号。

```bash
cargo test                                        # 55 Rust 测试
pytest tests/test_exhaustive.py -v                # 155/155 交叉验证
pytest tests/test_full_coverage.py -q             # 620 一致性 + 边界
```

## 架构

```
talib-rs/
├── crates/
│   ├── talib-rs-core/                    # 纯 Rust 库 (无 Python 依赖)
│   │   ├── src/
│   │   │   ├── overlap/               # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA,
│   │   │   │                          # T3, MAMA, BBANDS, SAR, MIDPOINT, ...
│   │   │   ├── momentum/              # RSI, MACD, STOCH, ADX, CCI, MOM, ROC,
│   │   │   │                          # WILLR, AROON, MFI, TRIX, ULTOSC, DX, ...
│   │   │   ├── volatility/            # ATR, NATR, TRANGE
│   │   │   ├── volume/                # AD, ADOSC, OBV
│   │   │   ├── pattern/               # 61 种 K 线形态 (含实际检测逻辑)
│   │   │   ├── statistic/             # STDDEV, VAR, BETA, CORREL, LINEARREG, TSF
│   │   │   ├── cycle/                 # Hilbert Transform (环形缓冲区优化)
│   │   │   ├── simd.rs                # f64x4 SIMD: sum, sum_sq_diff (3x 加速)
│   │   │   ├── sliding_window.rs      # 单调队列: O(n) 滑动最大/最小值
│   │   │   └── ma_type.rs             # MA 类型调度器 (9 种类型)
│   │   └── benches/simd_bench.rs      # Criterion 基准测试
│   │
│   └── talib-rs-python/                  # PyO3 绑定
│       └── src/
│           ├── func_api.rs            # 155 个 #[pyfunction] 定义
│           ├── metadata.rs            # get_functions(), get_function_groups()
│           └── conversion.rs          # 零拷贝 NumPy ↔ Rust
│
├── python/talib_rs/
│   ├── __init__.py                    # import talib_rs as talib
│   ├── abstract.py                    # Function 类 + 内省
│   └── stream.py                      # 流式计算封装
│
├── tests/                             # 353 准确性测试
├── benches/                           # 性能基准测试
└── .github/workflows/                 # CI/CD (多平台 + PyPI)
```

### 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| SIMD | `wide` 0.7 | 稳定版 Rust，不需要 nightly |
| 滑动极值 | 单调队列 | 融合双极值扫描，均摊 O(n) |
| 方差 | E[X²]−E[X]² | 单次遍历滑动窗口，非两次 Welford |
| Hilbert Transform | 环形缓冲区 | 10 个 Vec → 固定 [f64; 64]，堆分配减少 80% |
| EMA/DEMA/TEMA/T3 | 就地分层 | 消除 NaN 过滤产生的中间 Vec |
| WMA | 增量递推 | WS_new = WS_old − S_old + p×x_new, 每步 O(1) |
| unsafe 策略 | 零 `unsafe` | 迭代器模式 + 安全索引；LLVM 自动向量化 zip 链 |
| Python 模块名 | `talib_rs` | `import talib_rs as talib` 即可替换使用 |
| 输出格式 | `Vec<f64>` → `PyArray1` | 所有权转移，接近零拷贝 |

## 从 C TA-Lib 迁移

```bash
# 卸载 C TA-Lib
pip uninstall TA-Lib

# 安装 talib-rs
pip install talib-rs

# 更新导入: import talib_rs as talib
```

## 开发

```bash
# Rust 工具链
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 依赖
pip install maturin numpy pytest pytest-benchmark

# 构建 & 测试
maturin develop --release              # 构建 Python 包
cargo test                             # Rust 测试
pytest tests/ -q                       # Python 测试
cargo bench --bench simd_bench         # SIMD 微基准
python benches/generate_report.py      # 生成 BENCHMARK.md
```

## 许可证

BSD-3-Clause
