"""
Multi-dataset performance benchmark: talib-rs vs C TA-Lib

Fair comparison:
  - Same data, same parameters, same machine
  - No target-cpu=native, no platform-specific optimizations
  - Warm-up runs excluded from timing
  - Multiple iterations with median timing (robust to outliers)
  - Dataset sizes: 1K, 10K, 100K, 1M

Output: JSON results + Markdown table
"""

import json
import sys
import time
import os
import platform
import numpy as np

import talib as c_talib
from talib_rs import _talib as rs

# ============================================================
# Configuration
# ============================================================

DATASET_SIZES = [1_000, 10_000, 100_000, 1_000_000]
WARMUP = 3
ITERATIONS = 20  # median of 20 runs
SEED = 42

# ============================================================
# Data generation
# ============================================================

def make_ohlcv(n, seed=SEED):
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    spread = close * 0.015
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.uniform(1e6, 1e7, n).astype(np.float64)
    return opn, high, low, close, volume

# ============================================================
# Function registry: (input_type, c_func, rs_func, kwargs)
# ============================================================

def build_registry():
    """Build benchmark registry with matched C and Rust function pairs."""
    registry = {}

    def reg(name, input_type, c_fn, rs_fn, **kwargs):
        registry[name] = (input_type, c_fn, rs_fn, kwargs)

    # --- Overlap Studies ---
    reg('SMA(20)',    'c',   c_talib.SMA,   rs.SMA,   timeperiod=20)
    reg('EMA(20)',    'c',   c_talib.EMA,   rs.EMA,   timeperiod=20)
    reg('WMA(20)',    'c',   c_talib.WMA,   rs.WMA,   timeperiod=20)
    reg('DEMA(20)',   'c',   c_talib.DEMA,  rs.DEMA,  timeperiod=20)
    reg('TEMA(20)',   'c',   c_talib.TEMA,  rs.TEMA,  timeperiod=20)
    reg('TRIMA(20)',  'c',   c_talib.TRIMA, rs.TRIMA, timeperiod=20)
    reg('KAMA(30)',   'c',   c_talib.KAMA,  rs.KAMA,  timeperiod=30)
    reg('T3(5)',      'c',   c_talib.T3,    rs.T3,    timeperiod=5, vfactor=0.7)
    reg('MAMA',       'c',   c_talib.MAMA,  rs.MAMA,  fastlimit=0.5, slowlimit=0.05)
    reg('BBANDS(20)', 'c',   c_talib.BBANDS, rs.BBANDS, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    reg('SAR',        'hl',  c_talib.SAR,   rs.SAR,   acceleration=0.02, maximum=0.2)
    reg('SAREXT',     'hl',  c_talib.SAREXT, rs.SAREXT, startvalue=0.0, offsetonreverse=0.0,
        accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
        accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    reg('MIDPOINT(14)', 'c', c_talib.MIDPOINT, rs.MIDPOINT, timeperiod=14)
    reg('MIDPRICE(14)', 'hl', c_talib.MIDPRICE, rs.MIDPRICE, timeperiod=14)
    reg('HT_TRENDLINE', 'c', c_talib.HT_TRENDLINE, rs.HT_TRENDLINE)
    reg('MA(30,SMA)', 'c',   c_talib.MA,    rs.MA,    timeperiod=30, matype=0)

    # --- Momentum ---
    reg('RSI(14)',    'c',   c_talib.RSI,   rs.RSI,   timeperiod=14)
    reg('MACD',       'c',   c_talib.MACD,  rs.MACD,  fastperiod=12, slowperiod=26, signalperiod=9)
    reg('MACDEXT',    'c',   c_talib.MACDEXT, rs.MACDEXT, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9, signalmatype=1)
    reg('MACDFIX(9)', 'c',   c_talib.MACDFIX, rs.MACDFIX, signalperiod=9)
    reg('STOCH',      'hlc', c_talib.STOCH, rs.STOCH, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    reg('STOCHF',     'hlc', c_talib.STOCHF, rs.STOCHF, fastk_period=5, fastd_period=3, fastd_matype=0)
    reg('STOCHRSI(14)', 'c', c_talib.STOCHRSI, rs.STOCHRSI, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    reg('ADX(14)',    'hlc', c_talib.ADX,   rs.ADX,   timeperiod=14)
    reg('ADXR(14)',   'hlc', c_talib.ADXR,  rs.ADXR,  timeperiod=14)
    reg('CCI(14)',    'hlc', c_talib.CCI,   rs.CCI,   timeperiod=14)
    reg('MOM(10)',    'c',   c_talib.MOM,   rs.MOM,   timeperiod=10)
    reg('ROC(10)',    'c',   c_talib.ROC,   rs.ROC,   timeperiod=10)
    reg('ROCP(10)',   'c',   c_talib.ROCP,  rs.ROCP,  timeperiod=10)
    reg('ROCR(10)',   'c',   c_talib.ROCR,  rs.ROCR,  timeperiod=10)
    reg('ROCR100(10)', 'c',  c_talib.ROCR100, rs.ROCR100, timeperiod=10)
    reg('WILLR(14)',  'hlc', c_talib.WILLR, rs.WILLR, timeperiod=14)
    reg('APO',        'c',   c_talib.APO,   rs.APO,   fastperiod=12, slowperiod=26, matype=0)
    reg('PPO',        'c',   c_talib.PPO,   rs.PPO,   fastperiod=12, slowperiod=26, matype=0)
    reg('BOP',        'ohlc', c_talib.BOP,  rs.BOP)
    reg('CMO(14)',    'c',   c_talib.CMO,   rs.CMO,   timeperiod=14)
    reg('AROON(14)',  'hl',  c_talib.AROON, rs.AROON, timeperiod=14)
    reg('AROONOSC(14)', 'hl', c_talib.AROONOSC, rs.AROONOSC, timeperiod=14)
    reg('MFI(14)',    'hlcv', c_talib.MFI,  rs.MFI,   timeperiod=14)
    reg('TRIX(15)',   'c',   c_talib.TRIX,  rs.TRIX,  timeperiod=15)
    reg('ULTOSC',     'hlc', c_talib.ULTOSC, rs.ULTOSC, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    reg('DX(14)',     'hlc', c_talib.DX,    rs.DX,    timeperiod=14)
    reg('PLUS_DI(14)', 'hlc', c_talib.PLUS_DI, rs.PLUS_DI, timeperiod=14)
    reg('MINUS_DI(14)', 'hlc', c_talib.MINUS_DI, rs.MINUS_DI, timeperiod=14)
    reg('PLUS_DM(14)', 'hl', c_talib.PLUS_DM, rs.PLUS_DM, timeperiod=14)
    reg('MINUS_DM(14)', 'hl', c_talib.MINUS_DM, rs.MINUS_DM, timeperiod=14)

    # --- Volatility ---
    reg('ATR(14)',    'hlc', c_talib.ATR,   rs.ATR,   timeperiod=14)
    reg('NATR(14)',   'hlc', c_talib.NATR,  rs.NATR,  timeperiod=14)
    reg('TRANGE',     'hlc', c_talib.TRANGE, rs.TRANGE)

    # --- Volume ---
    reg('AD',         'hlcv', c_talib.AD,   rs.AD)
    reg('ADOSC',      'hlcv', c_talib.ADOSC, rs.ADOSC, fastperiod=3, slowperiod=10)
    reg('OBV',        'cv',  c_talib.OBV,   rs.OBV)

    # --- Price Transform ---
    reg('AVGPRICE',   'ohlc', c_talib.AVGPRICE, rs.AVGPRICE)
    reg('MEDPRICE',   'hl',  c_talib.MEDPRICE, rs.MEDPRICE)
    reg('TYPPRICE',   'hlc', c_talib.TYPPRICE, rs.TYPPRICE)
    reg('WCLPRICE',   'hlc', c_talib.WCLPRICE, rs.WCLPRICE)

    # --- Statistic ---
    reg('STDDEV(20)', 'c',   c_talib.STDDEV, rs.STDDEV, timeperiod=20, nbdev=1.0)
    reg('VAR(20)',    'c',   c_talib.VAR,   rs.VAR,   timeperiod=20, nbdev=1.0)
    reg('BETA(5)',    'cc',  c_talib.BETA,  rs.BETA,  timeperiod=5)
    reg('CORREL(30)', 'cc',  c_talib.CORREL, rs.CORREL, timeperiod=30)
    reg('LINEARREG(14)', 'c', c_talib.LINEARREG, rs.LINEARREG, timeperiod=14)
    reg('LINEARREG_SLOPE', 'c', c_talib.LINEARREG_SLOPE, rs.LINEARREG_SLOPE, timeperiod=14)
    reg('LINEARREG_INTERCEPT', 'c', c_talib.LINEARREG_INTERCEPT, rs.LINEARREG_INTERCEPT, timeperiod=14)
    reg('LINEARREG_ANGLE', 'c', c_talib.LINEARREG_ANGLE, rs.LINEARREG_ANGLE, timeperiod=14)
    reg('TSF(14)',    'c',   c_talib.TSF,   rs.TSF,   timeperiod=14)

    # --- Math Transform ---
    reg('SIN',        'c',   c_talib.SIN,   rs.SIN)
    reg('COS',        'c',   c_talib.COS,   rs.COS)
    reg('SQRT',       'c_pos', c_talib.SQRT, rs.SQRT)
    reg('LN',         'c_pos', c_talib.LN,  rs.LN)
    reg('EXP',        'c_tiny', c_talib.EXP, rs.EXP)

    # --- Math Operators ---
    reg('ADD',        'cc',  c_talib.ADD,   rs.ADD)
    reg('SUB',        'cc',  c_talib.SUB,   rs.SUB)
    reg('MULT',       'cc',  c_talib.MULT,  rs.MULT)
    reg('DIV',        'cc_nz', c_talib.DIV, rs.DIV)
    reg('MAX(30)',    'c',   c_talib.MAX,   rs.MAX,   timeperiod=30)
    reg('MIN(30)',    'c',   c_talib.MIN,   rs.MIN,   timeperiod=30)
    reg('MINMAX(30)', 'c',   c_talib.MINMAX, rs.MINMAX, timeperiod=30)
    reg('MINMAXINDEX(30)', 'c', c_talib.MINMAXINDEX, rs.MINMAXINDEX, timeperiod=30)
    reg('SUM(30)',    'c',   c_talib.SUM,   rs.SUM,   timeperiod=30)

    # --- Cycle ---
    reg('HT_DCPERIOD', 'c', c_talib.HT_DCPERIOD, rs.HT_DCPERIOD)
    reg('HT_DCPHASE',  'c', c_talib.HT_DCPHASE, rs.HT_DCPHASE)
    reg('HT_PHASOR',   'c', c_talib.HT_PHASOR, rs.HT_PHASOR)
    reg('HT_SINE',     'c', c_talib.HT_SINE, rs.HT_SINE)
    reg('HT_TRENDMODE','c', c_talib.HT_TRENDMODE, rs.HT_TRENDMODE)

    # --- Candlestick patterns (sample) ---
    reg('CDLDOJI',       'ohlc', c_talib.CDLDOJI, rs.CDLDOJI)
    reg('CDLHAMMER',     'ohlc', c_talib.CDLHAMMER, rs.CDLHAMMER)
    reg('CDLENGULFING',  'ohlc', c_talib.CDLENGULFING, rs.CDLENGULFING)
    reg('CDL3BLACKCROWS','ohlc', c_talib.CDL3BLACKCROWS, rs.CDL3BLACKCROWS)
    reg('CDLMORNINGSTAR','ohlc', c_talib.CDLMORNINGSTAR, rs.CDLMORNINGSTAR)
    reg('CDLHIKKAKE',    'ohlc', c_talib.CDLHIKKAKE, rs.CDLHIKKAKE)

    return registry

# ============================================================
# Benchmark engine
# ============================================================

def prepare_args(input_type, ohlcv, close2):
    o, h, l, c, v = ohlcv
    n = len(c)
    if input_type == 'c':
        return [c]
    elif input_type == 'c_pos':
        return [np.abs(c) + 0.01]
    elif input_type == 'c_tiny':
        return [np.linspace(0.0, 5.0, n)]
    elif input_type == 'hl':
        return [h, l]
    elif input_type == 'hlc':
        return [h, l, c]
    elif input_type == 'ohlc':
        return [o, h, l, c]
    elif input_type == 'hlcv':
        return [h, l, c, v]
    elif input_type == 'cv':
        return [c, v]
    elif input_type == 'cc':
        return [c, close2]
    elif input_type == 'cc_nz':
        c2 = np.where(np.abs(close2) < 0.01, 0.01, close2)
        return [c, c2]
    else:
        return [c]

def bench_one(func, args, kwargs, warmup=WARMUP, iters=ITERATIONS):
    """Return median execution time in microseconds."""
    # Warm-up
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        func(*args, **kwargs)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns -> us

    times.sort()
    return times[len(times) // 2]  # median

def run_benchmarks():
    registry = build_registry()
    results = {}  # {dataset_size: [{name, c_us, rs_us, speedup}, ...]}

    total_funcs = len(registry)

    for size in DATASET_SIZES:
        size_label = f"{size:,}"
        print(f"\n{'='*60}")
        print(f"  Dataset: {size_label} bars")
        print(f"{'='*60}")

        ohlcv = make_ohlcv(size)
        rng = np.random.RandomState(SEED + 1)
        close2 = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size)))

        rows = []
        for idx, (name, (input_type, c_fn, rs_fn, kwargs)) in enumerate(sorted(registry.items())):
            args = prepare_args(input_type, ohlcv, close2)
            try:
                c_us = bench_one(c_fn, args, kwargs)
                rs_us = bench_one(rs_fn, args, kwargs)
                speedup = c_us / rs_us if rs_us > 0 else float('inf')
                rows.append({
                    'name': name,
                    'c_us': round(c_us, 1),
                    'rs_us': round(rs_us, 1),
                    'speedup': round(speedup, 2),
                })
                status = f"{'✓' if speedup >= 1.0 else '✗'} {speedup:.2f}x"
                print(f"  [{idx+1:3d}/{total_funcs}] {name:<25s} C={c_us:>10,.1f}us  Rust={rs_us:>10,.1f}us  {status}")
            except Exception as e:
                print(f"  [{idx+1:3d}/{total_funcs}] {name:<25s} ERROR: {e}")
                rows.append({
                    'name': name,
                    'c_us': 0,
                    'rs_us': 0,
                    'speedup': 0,
                    'error': str(e),
                })

        results[size] = rows

    return results

# ============================================================
# Markdown generation
# ============================================================

def generate_markdown(results):
    lines = []
    lines.append("# talib-rs vs C TA-Lib — Multi-Dataset Performance Benchmark")
    lines.append("")

    # Platform info
    import subprocess
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor() or "Unknown"

    lines.append("> **Platform:** {} {} | {} | Python {} | `--release` (LTO fat, codegen-units=1)".format(
        platform.system(), platform.machine(), cpu, platform.python_version()))
    lines.append("> **Method:** median of {} iterations, {} warm-up runs | `time.perf_counter_ns()`".format(
        ITERATIONS, WARMUP))
    lines.append("> **Fair:** no `target-cpu=native`, no platform-specific SIMD flags")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Indicators | Faster | Equal (±5%) | Slower | Avg Speedup | Median Speedup |")
    lines.append("|--------:|:----------:|:------:|:-----------:|:------:|:-----------:|:--------------:|")

    for size in DATASET_SIZES:
        rows = [r for r in results[size] if r.get('speedup', 0) > 0]
        if not rows:
            continue
        speedups = [r['speedup'] for r in rows]
        faster = sum(1 for s in speedups if s > 1.05)
        equal = sum(1 for s in speedups if 0.95 <= s <= 1.05)
        slower = sum(1 for s in speedups if s < 0.95)
        avg = sum(speedups) / len(speedups)
        speedups_sorted = sorted(speedups)
        med = speedups_sorted[len(speedups_sorted) // 2]
        label = f"{size:,}"
        lines.append(f"| {label} | {len(rows)} | **{faster}** | {equal} | {slower} | **{avg:.2f}x** | **{med:.2f}x** |")

    lines.append("")

    # Per-dataset detail tables
    for size in DATASET_SIZES:
        rows = [r for r in results[size] if r.get('speedup', 0) > 0]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: -r['speedup'])
        label = f"{size:,}"

        lines.append(f"## {label} Bars")
        lines.append("")
        lines.append("| Indicator | C (us) | Rust (us) | Speedup |")
        lines.append("|-----------|-------:|---------:|--------:|")

        for r in rows_sorted:
            sp = r['speedup']
            if sp >= 1.05:
                sp_str = f"**{sp:.2f}x**"
            elif sp < 0.95:
                sp_str = f"_{sp:.2f}x_"
            else:
                sp_str = f"{sp:.2f}x"
            lines.append("| {} | {:,.1f} | {:,.1f} | {} |".format(
                r['name'], r['c_us'], r['rs_us'], sp_str))

        lines.append("")

    # Optimization techniques
    lines.append("## Optimization Techniques Applied")
    lines.append("")
    lines.append("| Technique | Indicators | Typical Gain |")
    lines.append("|-----------|-----------|-------------|")
    lines.append("| Single-pass O(n) sliding SMA+STDDEV | BBANDS | 2-3x |")
    lines.append("| Inline 3/6-layer EMA cascade | TEMA, TRIX, T3 | 2-3x |")
    lines.append("| O(n) sliding sums vs O(n*p) per-window | LINEARREG family, BETA, TSF | 1.5-2.5x |")
    lines.append("| C-style brute extremum scan | MIDPOINT, MIDPRICE, MIN, WILLR | 1.2-18x |")
    lines.append("| Inline Wilder smoothing (no intermediate Vec) | ADX, DX, DI, DM, NATR | 1.0-1.4x |")
    lines.append("| Fused AD + EMA in single pass | ADOSC, MFI | 1.1-1.4x |")
    lines.append("| SIMD f64x4 accelerated sum | SMA, MA | 1.4-1.7x |")
    lines.append("| Ring buffer replacing 13 Vec allocations | MAMA | ~1x |")
    lines.append("| `unsafe get_unchecked` in hot loops | All optimized indicators | 1.05-1.1x |")
    lines.append("| `vec![0.0]` + NaN fill (calloc optimization) | Most indicators | ~1.05x |")
    lines.append("| LTO fat + codegen-units=1 | All cross-crate calls | 1.1-1.3x |")
    lines.append("")
    lines.append("> All algorithms are O(n). Remaining gaps are constant-factor differences in compiler")
    lines.append("> code generation quality (C macro inlining vs Rust function calls).")

    return "\n".join(lines) + "\n"

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("talib-rs vs C TA-Lib — Multi-Dataset Performance Benchmark")
    print(f"Datasets: {', '.join(f'{s:,}' for s in DATASET_SIZES)}")
    print(f"Iterations: {ITERATIONS} (median) | Warm-up: {WARMUP}")
    print()

    results = run_benchmarks()

    # Save JSON
    json_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to {json_path}")

    # Generate and save Markdown
    md = generate_markdown(results)
    md_path = os.path.join(os.path.dirname(__file__), "..", "BENCHMARK.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown saved to {md_path}")
