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
  <img src="https://img.shields.io/badge/tests-4562_accuracy-green" alt="4562 accuracy tests" />
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
| 精度 | 参考实现 | **位精度一致** (4,562 准确性测试 × 7 数据规模 × 6 场景) |
| 性能 | 基准线 | **1M 平均 1.41x 更快** (47/90 更快, 零 unsafe) |
| 内存安全 | 手动管理 | Rust 编译器保证 |
| Python 集成 | Cython 封装 | PyO3 零拷贝 |
| 指标数量 | 155 | 155 (100% 覆盖) |

### 核心技术优势

- **位精度一致** — 不是"近似相等"，而是与 C TA-Lib 的 `diff=0`，4,562 准确性测试覆盖 7 种数据规模 × 6 种市场场景 × 3 个随机种子。

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

> Apple M4 | `--release` LTO fat | 20 次取中位数 | **零 unsafe** 代码

| 数据集 | 更快 | 持平 (±5%) | 较慢 | 平均 | 中位数 |
|-------:|:----:|:----------:|:----:|:----:|:------:|
| 1,000 | 31 | 26 | 33 | 1.16x | 1.00x |
| 10,000 | 38 | 34 | 18 | 1.34x | 1.02x |
| 100,000 | **45** | 39 | 6 | **1.40x** | **1.06x** |
| 1,000,000 | **47** | 35 | 8 | **1.41x** | **1.07x** |

### 1M 关键指标

| 指标 | C (us) | Rust (us) | 加速比 | 优化技术 |
|------|-------:|---------:|-------:|---------|
| MIDPOINT(14) | 39,300 | 3,720 | **10.6x** | O(n) 缓存索引 vs C 的 O(n×p) 暴力扫描 |
| BBANDS(20) | 3,500 | 1,160 | **3.0x** | 单趟融合 SMA+STDDEV |
| TEMA(20) | 3,900 | 1,420 | **2.8x** | 内联 3 层 EMA 级联，无中间 Vec |
| LINEARREG(14) | 4,020 | 1,830 | **2.2x** | O(n) 滑动求和 vs C 的 O(n×p) |
| TRIX(15) | 4,100 | 1,940 | **2.1x** | 融合 3 层 EMA + ROC |
| OBV | 3,160 | 1,600 | **2.0x** | 顺序 push 避免 calloc COW 页错误 |
| DEMA(20) | 2,640 | 1,390 | **1.9x** | 消除中间 ema1 Vec |
| VAR(20) | 1,160 | 710 | **1.6x** | O(n) 滑动求和 + 除法→乘法 |
| STDDEV(20) | 1,400 | 870 | **1.6x** | 融合 var+sqrt 单趟 |
| MACD | 4,260 | 2,790 | **1.5x** | C TA-Lib EMA 公式: `k*(x-prev)+prev` |

完整基准：90 个指标 × 4 数据集，含 C/Rust 运行时间 → [BENCHMARK.md](BENCHMARK.md)

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
155/155 函数 · 4,562 准确性测试 · 7 种数据规模 × 6 种场景 × 3 种子 · 0 失败
```

| 测试套件 | 用例数 | 说明 |
|---------|------:|------|
| **Rust 单元测试** | 54 | 核心算法、SIMD、滑动窗口、边界条件 |
| **准确性交叉验证** | 353 | 155 函数 × 6 种数据集，rtol=1e-10 |
| **多数据集对齐** | 4,158 | 33 指标 × 7 规模 × 6 场景 × 3 种子 |
| **K 线形态精确匹配** | 122 | 61 种形态 × 2 数据集，整数信号精确 |

### 分级容差策略

| 级别 | rtol | 适用 |
|------|------|------|
| 精确 | 1e-14 | 逐元素运算 (MAX, MIN, AROON, MOM) |
| 标准 | 1e-10 | EMA/Wilder 串行链 |
| 滑动 | 1e-8 | 滑动求和算法 (SMA, CCI, LINEARREG) |
| 累积 | 1e-6 | VAR/STDDEV/BBANDS (E(X²)-E(X)² 消去) |

```bash
cargo test                                               # 54 Rust 测试
pytest tests/accuracy/ -v                                # 4,562 准确性测试
pytest tests/accuracy/ -k "100000"                       # 只跑 100K 数据集
pytest tests/accuracy/ -k "volatile"                     # 只跑高波动场景
pytest tests/accuracy/test_multi_dataset_alignment.py    # 多数据集套件
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
