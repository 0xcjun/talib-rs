# talib-rs vs C TA-Lib — Performance Benchmark

> **Platform:** Darwin arm64 | Apple M4 | Python 3.13.5 | `--release` (LTO fat, codegen-units=1)
> **Method:** median of 20 iterations, 3 warm-up runs | `time.perf_counter_ns()`
> **Fair:** no `target-cpu=native`, no platform-specific SIMD flags, **zero unsafe** code

## Summary

| Dataset | Indicators | Faster | Equal (±5%) | Slower | Avg Speedup | Median Speedup |
|--------:|:----------:|:------:|:-----------:|:------:|:-----------:|:--------------:|
| 1,000 | 90 | **27** | 14 | 49 | **1.09x** | **0.94x** |
| 10,000 | 90 | **34** | 22 | 34 | **1.25x** | **0.97x** |
| 100,000 | 90 | **43** | 28 | 19 | **1.35x** | **1.04x** |
| 1,000,000 | 90 | **39** | 31 | 20 | **1.33x** | **1.02x** |

## All Results (sorted by 1M speedup)

| Indicator | 1K | 10K | 100K | 1M |
|-----------|---:|----:|-----:|---:|
| MIDPOINT(14) | **3.49x** | **13.34x** | **12.07x** | **10.67x** |
| ROCR(10) | **1.50x** | **2.29x** | **2.87x** | **3.04x** |
| BBANDS(20) | **1.71x** | **2.51x** | **3.22x** | **2.88x** |
| LINEARREG_INTERCEPT | **2.43x** | **2.26x** | **2.50x** | **2.45x** |
| TEMA(20) | **2.09x** | **2.39x** | **2.41x** | **2.41x** |
| LINEARREG_SLOPE | **2.20x** | **2.27x** | **2.36x** | **2.34x** |
| LINEARREG(14) | **1.78x** | **1.80x** | **2.04x** | **2.16x** |
| TSF(14) | **1.88x** | **1.88x** | **2.08x** | **2.13x** |
| TRIX(15) | **1.74x** | **1.93x** | **1.94x** | **1.93x** |
| OBV | _0.43x_ | _0.41x_ | **1.77x** | **1.92x** |
| STOCH | **1.19x** | _0.89x_ | **1.72x** | **1.85x** |
| LINEARREG_ANGLE | **2.20x** | **1.79x** | **1.76x** | **1.84x** |
| BETA(5) | **1.60x** | **1.56x** | **1.79x** | **1.71x** |
| ADX(14) | 1.03x | **1.08x** | **1.40x** | **1.62x** |
| STOCHF | _0.74x_ | _0.77x_ | **1.42x** | **1.62x** |
| MA(30,SMA) | **1.19x** | **1.53x** | **1.66x** | **1.60x** |
| MFI(14) | _0.50x_ | _0.45x_ | **1.55x** | **1.57x** |
| SMA(20) | **1.30x** | **1.48x** | **1.65x** | **1.55x** |
| MIN(30) | **1.39x** | **1.47x** | **1.54x** | **1.54x** |
| TRIMA(20) | **1.36x** | **1.35x** | **1.53x** | **1.46x** |
| ADXR(14) | _0.93x_ | _0.94x_ | **1.21x** | **1.43x** |
| SUM(30) | **1.22x** | **1.41x** | **1.50x** | **1.43x** |
| CCI(14) | **1.29x** | **1.37x** | **1.26x** | **1.40x** |
| CDLHAMMER | **1.16x** | _0.84x_ | **1.26x** | **1.37x** |
| MACDFIX(9) | **1.18x** | **1.28x** | **1.31x** | **1.35x** |
| MACD | **1.19x** | **1.29x** | **1.31x** | **1.33x** |
| MIDPRICE(14) | **2.14x** | **1.88x** | **1.36x** | **1.30x** |
| WILLR(14) | 0.97x | _0.85x_ | **1.20x** | **1.26x** |
| PPO | **1.11x** | **1.11x** | **1.20x** | **1.19x** |
| CORREL(30) | **1.06x** | **1.13x** | **1.13x** | **1.18x** |
| BOP | _0.94x_ | **1.12x** | **1.27x** | **1.17x** |
| PLUS_DI(14) | **1.12x** | **1.17x** | 1.04x | **1.17x** |
| CDL3BLACKCROWS | **1.49x** | **1.11x** | **1.16x** | **1.16x** |
| APO | 1.00x | **1.05x** | **1.12x** | **1.14x** |
| DX(14) | _0.82x_ | _0.78x_ | **1.86x** | **1.14x** |
| CDLDOJI | 0.96x | **1.18x** | **1.16x** | **1.12x** |
| CMO(14) | 1.02x | **1.05x** | **1.12x** | **1.11x** |
| AD | _0.88x_ | **1.14x** | **1.10x** | **1.08x** |
| ROC(10) | _0.94x_ | **1.06x** | **1.07x** | **1.07x** |
| NATR(14) | 1.00x | **1.06x** | **1.05x** | **1.05x** |
| STOCHRSI(14) | _0.80x_ | _0.83x_ | **1.17x** | **1.05x** |
| ULTOSC | 1.02x | 0.97x | **1.07x** | **1.05x** |
| MAX(30) | _0.93x_ | _0.83x_ | 1.01x | 1.04x |
| MINMAXINDEX(30) | 1.03x | 1.00x | 1.03x | 1.02x |
| PLUS_DM(14) | _0.91x_ | _0.94x_ | 0.95x | 1.02x |
| RSI(14) | 1.00x | 1.02x | 1.02x | 1.02x |
| ROCR100(10) | _0.94x_ | 0.99x | 1.00x | 1.01x |
| ADD | _0.75x_ | **1.07x** | 1.00x | 1.00x |
| COS | 1.02x | 0.97x | _0.88x_ | 1.00x |
| HT_SINE | 1.01x | 1.01x | 1.01x | 1.00x |
| MINUS_DM(14) | _0.91x_ | _0.94x_ | 1.00x | 1.00x |
| ROCP(10) | _0.88x_ | 0.99x | 1.00x | 1.00x |
| WCLPRICE | _0.70x_ | 0.99x | 1.00x | 1.00x |
| DIV | _0.87x_ | 0.97x | 0.99x | 0.99x |
| MEDPRICE | _0.75x_ | **1.39x** | 1.00x | 0.99x |
| MOM(10) | _0.89x_ | 0.97x | 1.00x | 0.99x |
| SIN | 0.98x | **1.21x** | **1.07x** | 0.99x |
| SUB | _0.78x_ | 0.97x | 1.00x | 0.99x |
| TRANGE | _0.82x_ | 0.96x | 0.99x | 0.99x |
| TYPPRICE | _0.70x_ | 0.96x | 1.00x | 0.99x |
| MAMA | _0.83x_ | _0.85x_ | 0.99x | 0.98x |
| MINUS_DI(14) | _0.92x_ | 0.98x | 0.99x | 0.98x |
| HT_DCPHASE | 0.98x | 0.96x | 0.96x | 0.97x |
| LN | _0.93x_ | _0.83x_ | 0.96x | 0.97x |
| ADOSC | _0.86x_ | _0.94x_ | 0.95x | 0.96x |
| CDLENGULFING | _0.92x_ | 0.96x | _0.69x_ | 0.96x |
| CDLMORNINGSTAR | **1.57x** | **1.55x** | _0.90x_ | 0.96x |
| WMA(20) | _0.94x_ | 1.00x | 0.99x | 0.96x |
| AVGPRICE | _0.73x_ | 0.95x | 1.00x | 0.95x |
| HT_TRENDMODE | 0.95x | 0.96x | 0.96x | 0.95x |
| EXP | _0.90x_ | _0.90x_ | _0.87x_ | _0.94x_ |
| MULT | _0.89x_ | 0.95x | 0.99x | _0.94x_ |
| VAR(20) | _0.91x_ | _0.92x_ | 0.96x | _0.94x_ |
| HT_PHASOR | _0.94x_ | _0.93x_ | _0.93x_ | _0.93x_ |
| STDDEV(20) | _0.84x_ | _0.87x_ | _0.92x_ | _0.93x_ |
| HT_TRENDLINE | _0.92x_ | _0.92x_ | _0.92x_ | _0.92x_ |
| HT_DCPERIOD | _0.90x_ | _0.91x_ | _0.91x_ | _0.90x_ |
| SAR | _0.80x_ | _0.91x_ | **1.22x** | _0.89x_ |
| DEMA(20) | _0.82x_ | _0.86x_ | _0.88x_ | _0.87x_ |
| AROON(14) | _0.76x_ | _0.80x_ | _0.77x_ | _0.85x_ |
| AROONOSC(14) | _0.74x_ | _0.72x_ | _0.77x_ | _0.85x_ |
| SAREXT | _0.74x_ | _0.77x_ | _0.71x_ | _0.84x_ |
| SQRT | _0.75x_ | _0.78x_ | _0.80x_ | _0.82x_ |
| T3(5) | _0.80x_ | _0.81x_ | _0.81x_ | _0.82x_ |
| CDLHIKKAKE | _0.67x_ | _0.78x_ | _0.94x_ | _0.81x_ |
| EMA(20) | _0.81x_ | _0.81x_ | **1.69x** | _0.81x_ |
| ATR(14) | _0.90x_ | _0.79x_ | _0.77x_ | _0.77x_ |
| MACDEXT | _0.74x_ | _0.74x_ | _0.75x_ | _0.75x_ |
| KAMA(30) | _0.69x_ | _0.72x_ | _0.71x_ | _0.74x_ |
| MINMAX(30) | _0.67x_ | _0.61x_ | _0.66x_ | _0.67x_ |

## Optimization Techniques

| Technique | Indicators | Typical Gain |
|-----------|-----------|-------------|
| Single-pass O(n) sliding SMA+STDDEV | BBANDS | 2-3x |
| Inline 3/6-layer EMA cascade | TEMA, TRIX, T3 | 2-3x |
| O(n) sliding sums vs O(n*p) per-window | LINEARREG family, BETA, TSF | 1.5-2.5x |
| C-style brute extremum scan | MIDPOINT, MIDPRICE, MIN, WILLR | 1.2-11x |
| `mul_add` hardware FMA | EMA, MACD, DEMA, TEMA, TRIX, ADOSC | ~7% |
| Slice iterator rescan (bounds elision) | MAX, MIN, AROON, WILLR, MINMAX | 10-40% |
| `Vec::with_capacity + extend` (no COW) | MOM, ROC, TRANGE, price transforms | 30-70% |
| `copysign` branchless pattern output | CDLDOJI, 57 CDL patterns | 2-3x |
| LTO fat + codegen-units=1 | All cross-crate calls | 1.1-1.3x |

> All algorithms are O(n). Zero `unsafe` blocks in entire codebase.
> Remaining gaps are serial data dependencies (EMA chains, Wilder smoothing)
> and memory bandwidth limits for trivially simple indicators.
