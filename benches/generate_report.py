#!/usr/bin/env python3
"""
全指标多数据集性能对比: talib-rs vs C TA-Lib
生成 Markdown 报告到 BENCHMARK.md

运行: python benches/generate_report.py
"""

import time
import sys
import numpy as np
from collections import OrderedDict
from pathlib import Path

import talib as c_talib
from talib import _talib as rs

# ============================================================
# 数据生成
# ============================================================

def make_ohlcv(n, seed=42):
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    spread = close * 0.015
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.uniform(1e6, 1e7, n)
    close2 = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.025, n)))
    periods = np.full(n, 10.0)
    return {
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume, 'close2': close2, 'periods': periods,
        'close_safe': np.linspace(-0.99, 0.99, n),
        'close_small': np.linspace(-2.0, 2.0, n),
        'close_tiny': np.linspace(0.01, 5.0, n),
        'close_pos': np.abs(close) + 0.01,
    }

SIZES = [1000, 10000, 100000]
DATA = {n: make_ohlcv(n, seed=42 + n) for n in SIZES}

# ============================================================
# 函数定义
# ============================================================

# (name, group, rs_call, c_call)
# rs_call / c_call: lambda data_dict -> callable with no args
FUNCS = []

def reg(name, group, build_rs, build_c):
    FUNCS.append((name, group, build_rs, build_c))

# --- Overlap Studies ---
for fn in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA']:
    reg(fn, 'Overlap', lambda d, f=fn: lambda: getattr(rs, f)(d['close'], 20),
                        lambda d, f=fn: lambda: getattr(c_talib, f)(d['close'], timeperiod=20))

reg('KAMA', 'Overlap', lambda d: lambda: rs.KAMA(d['close'], 30),
                        lambda d: lambda: c_talib.KAMA(d['close'], timeperiod=30))
reg('T3', 'Overlap', lambda d: lambda: rs.T3(d['close'], 5, 0.7),
                      lambda d: lambda: c_talib.T3(d['close'], timeperiod=5, vfactor=0.7))
reg('MAMA', 'Overlap', lambda d: lambda: rs.MAMA(d['close'], 0.5, 0.05),
                        lambda d: lambda: c_talib.MAMA(d['close'], fastlimit=0.5, slowlimit=0.05))
reg('BBANDS', 'Overlap', lambda d: lambda: rs.BBANDS(d['close'], 20, 2.0, 2.0, 0),
                          lambda d: lambda: c_talib.BBANDS(d['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0))
reg('SAR', 'Overlap', lambda d: lambda: rs.SAR(d['high'], d['low'], 0.02, 0.2),
                       lambda d: lambda: c_talib.SAR(d['high'], d['low'], acceleration=0.02, maximum=0.2))
reg('MIDPOINT', 'Overlap', lambda d: lambda: rs.MIDPOINT(d['close'], 14),
                            lambda d: lambda: c_talib.MIDPOINT(d['close'], timeperiod=14))
reg('MIDPRICE', 'Overlap', lambda d: lambda: rs.MIDPRICE(d['high'], d['low'], 14),
                            lambda d: lambda: c_talib.MIDPRICE(d['high'], d['low'], timeperiod=14))
reg('MAVP', 'Overlap', lambda d: lambda: rs.MAVP(d['close'], d['periods'], 2, 30, 0),
                        lambda d: lambda: c_talib.MAVP(d['close'], d['periods'], minperiod=2, maxperiod=30, matype=0))
reg('HT_TRENDLINE', 'Overlap', lambda d: lambda: rs.HT_TRENDLINE(d['close']),
                                lambda d: lambda: c_talib.HT_TRENDLINE(d['close']))

# --- Momentum ---
reg('RSI', 'Momentum', lambda d: lambda: rs.RSI(d['close'], 14), lambda d: lambda: c_talib.RSI(d['close'], timeperiod=14))
reg('MACD', 'Momentum', lambda d: lambda: rs.MACD(d['close'], 12, 26, 9), lambda d: lambda: c_talib.MACD(d['close'], fastperiod=12, slowperiod=26, signalperiod=9))
reg('MACDFIX', 'Momentum', lambda d: lambda: rs.MACDFIX(d['close'], 9), lambda d: lambda: c_talib.MACDFIX(d['close'], signalperiod=9))
reg('STOCH', 'Momentum', lambda d: lambda: rs.STOCH(d['high'], d['low'], d['close'], 5, 3, 0, 3, 0), lambda d: lambda: c_talib.STOCH(d['high'], d['low'], d['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0))
reg('STOCHF', 'Momentum', lambda d: lambda: rs.STOCHF(d['high'], d['low'], d['close'], 5, 3, 0), lambda d: lambda: c_talib.STOCHF(d['high'], d['low'], d['close'], fastk_period=5, fastd_period=3, fastd_matype=0))
reg('ADX', 'Momentum', lambda d: lambda: rs.ADX(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.ADX(d['high'], d['low'], d['close'], timeperiod=14))
reg('ADXR', 'Momentum', lambda d: lambda: rs.ADXR(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.ADXR(d['high'], d['low'], d['close'], timeperiod=14))
reg('CCI', 'Momentum', lambda d: lambda: rs.CCI(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.CCI(d['high'], d['low'], d['close'], timeperiod=14))
reg('MOM', 'Momentum', lambda d: lambda: rs.MOM(d['close'], 10), lambda d: lambda: c_talib.MOM(d['close'], timeperiod=10))
reg('ROC', 'Momentum', lambda d: lambda: rs.ROC(d['close'], 10), lambda d: lambda: c_talib.ROC(d['close'], timeperiod=10))
reg('WILLR', 'Momentum', lambda d: lambda: rs.WILLR(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.WILLR(d['high'], d['low'], d['close'], timeperiod=14))
reg('APO', 'Momentum', lambda d: lambda: rs.APO(d['close'], 12, 26, 0), lambda d: lambda: c_talib.APO(d['close'], fastperiod=12, slowperiod=26, matype=0))
reg('PPO', 'Momentum', lambda d: lambda: rs.PPO(d['close'], 12, 26, 0), lambda d: lambda: c_talib.PPO(d['close'], fastperiod=12, slowperiod=26, matype=0))
reg('BOP', 'Momentum', lambda d: lambda: rs.BOP(d['open'], d['high'], d['low'], d['close']), lambda d: lambda: c_talib.BOP(d['open'], d['high'], d['low'], d['close']))
reg('CMO', 'Momentum', lambda d: lambda: rs.CMO(d['close'], 14), lambda d: lambda: c_talib.CMO(d['close'], timeperiod=14))
reg('AROON', 'Momentum', lambda d: lambda: rs.AROON(d['high'], d['low'], 14), lambda d: lambda: c_talib.AROON(d['high'], d['low'], timeperiod=14))
reg('AROONOSC', 'Momentum', lambda d: lambda: rs.AROONOSC(d['high'], d['low'], 14), lambda d: lambda: c_talib.AROONOSC(d['high'], d['low'], timeperiod=14))
reg('MFI', 'Momentum', lambda d: lambda: rs.MFI(d['high'], d['low'], d['close'], d['volume'], 14), lambda d: lambda: c_talib.MFI(d['high'], d['low'], d['close'], d['volume'], timeperiod=14))
reg('TRIX', 'Momentum', lambda d: lambda: rs.TRIX(d['close'], 15), lambda d: lambda: c_talib.TRIX(d['close'], timeperiod=15))
reg('ULTOSC', 'Momentum', lambda d: lambda: rs.ULTOSC(d['high'], d['low'], d['close'], 7, 14, 28), lambda d: lambda: c_talib.ULTOSC(d['high'], d['low'], d['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28))
reg('DX', 'Momentum', lambda d: lambda: rs.DX(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.DX(d['high'], d['low'], d['close'], timeperiod=14))
reg('PLUS_DI', 'Momentum', lambda d: lambda: rs.PLUS_DI(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.PLUS_DI(d['high'], d['low'], d['close'], timeperiod=14))
reg('MINUS_DI', 'Momentum', lambda d: lambda: rs.MINUS_DI(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.MINUS_DI(d['high'], d['low'], d['close'], timeperiod=14))
reg('PLUS_DM', 'Momentum', lambda d: lambda: rs.PLUS_DM(d['high'], d['low'], 14), lambda d: lambda: c_talib.PLUS_DM(d['high'], d['low'], timeperiod=14))
reg('MINUS_DM', 'Momentum', lambda d: lambda: rs.MINUS_DM(d['high'], d['low'], 14), lambda d: lambda: c_talib.MINUS_DM(d['high'], d['low'], timeperiod=14))
reg('STOCHRSI', 'Momentum', lambda d: lambda: rs.STOCHRSI(d['close'], 14, 5, 3, 0), lambda d: lambda: c_talib.STOCHRSI(d['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0))

# --- Volatility ---
reg('ATR', 'Volatility', lambda d: lambda: rs.ATR(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.ATR(d['high'], d['low'], d['close'], timeperiod=14))
reg('NATR', 'Volatility', lambda d: lambda: rs.NATR(d['high'], d['low'], d['close'], 14), lambda d: lambda: c_talib.NATR(d['high'], d['low'], d['close'], timeperiod=14))
reg('TRANGE', 'Volatility', lambda d: lambda: rs.TRANGE(d['high'], d['low'], d['close']), lambda d: lambda: c_talib.TRANGE(d['high'], d['low'], d['close']))

# --- Volume ---
reg('AD', 'Volume', lambda d: lambda: rs.AD(d['high'], d['low'], d['close'], d['volume']), lambda d: lambda: c_talib.AD(d['high'], d['low'], d['close'], d['volume']))
reg('ADOSC', 'Volume', lambda d: lambda: rs.ADOSC(d['high'], d['low'], d['close'], d['volume'], 3, 10), lambda d: lambda: c_talib.ADOSC(d['high'], d['low'], d['close'], d['volume'], fastperiod=3, slowperiod=10))
reg('OBV', 'Volume', lambda d: lambda: rs.OBV(d['close'], d['volume']), lambda d: lambda: c_talib.OBV(d['close'], d['volume']))

# --- Price Transform ---
reg('AVGPRICE', 'PriceTransform', lambda d: lambda: rs.AVGPRICE(d['open'], d['high'], d['low'], d['close']), lambda d: lambda: c_talib.AVGPRICE(d['open'], d['high'], d['low'], d['close']))
reg('MEDPRICE', 'PriceTransform', lambda d: lambda: rs.MEDPRICE(d['high'], d['low']), lambda d: lambda: c_talib.MEDPRICE(d['high'], d['low']))
reg('TYPPRICE', 'PriceTransform', lambda d: lambda: rs.TYPPRICE(d['high'], d['low'], d['close']), lambda d: lambda: c_talib.TYPPRICE(d['high'], d['low'], d['close']))
reg('WCLPRICE', 'PriceTransform', lambda d: lambda: rs.WCLPRICE(d['high'], d['low'], d['close']), lambda d: lambda: c_talib.WCLPRICE(d['high'], d['low'], d['close']))

# --- Statistic ---
reg('STDDEV', 'Statistic', lambda d: lambda: rs.STDDEV(d['close'], 20, 1.0), lambda d: lambda: c_talib.STDDEV(d['close'], timeperiod=20, nbdev=1.0))
reg('VAR', 'Statistic', lambda d: lambda: rs.VAR(d['close'], 20, 1.0), lambda d: lambda: c_talib.VAR(d['close'], timeperiod=20, nbdev=1.0))
reg('BETA', 'Statistic', lambda d: lambda: rs.BETA(d['close'], d['close2'], 5), lambda d: lambda: c_talib.BETA(d['close'], d['close2'], timeperiod=5))
reg('CORREL', 'Statistic', lambda d: lambda: rs.CORREL(d['close'], d['close2'], 30), lambda d: lambda: c_talib.CORREL(d['close'], d['close2'], timeperiod=30))
reg('LINEARREG', 'Statistic', lambda d: lambda: rs.LINEARREG(d['close'], 14), lambda d: lambda: c_talib.LINEARREG(d['close'], timeperiod=14))
reg('LINEARREG_SLOPE', 'Statistic', lambda d: lambda: rs.LINEARREG_SLOPE(d['close'], 14), lambda d: lambda: c_talib.LINEARREG_SLOPE(d['close'], timeperiod=14))
reg('TSF', 'Statistic', lambda d: lambda: rs.TSF(d['close'], 14), lambda d: lambda: c_talib.TSF(d['close'], timeperiod=14))

# --- Math Transform ---
for fn in ['SIN', 'COS', 'TAN', 'SQRT', 'LN', 'LOG10', 'EXP', 'CEIL', 'FLOOR', 'TANH']:
    ck = 'close_pos' if fn in ('LN', 'LOG10', 'SQRT') else ('close_tiny' if fn == 'EXP' else 'close')
    reg(fn, 'MathTransform', lambda d, f=fn, k=ck: lambda: getattr(rs, f)(d[k]),
                              lambda d, f=fn, k=ck: lambda: getattr(c_talib, f)(d[k]))

# --- Math Operators ---
reg('ADD', 'MathOperator', lambda d: lambda: rs.ADD(d['close'], d['close2']), lambda d: lambda: c_talib.ADD(d['close'], d['close2']))
reg('SUB', 'MathOperator', lambda d: lambda: rs.SUB(d['close'], d['close2']), lambda d: lambda: c_talib.SUB(d['close'], d['close2']))
reg('MULT', 'MathOperator', lambda d: lambda: rs.MULT(d['close'], d['close2']), lambda d: lambda: c_talib.MULT(d['close'], d['close2']))
reg('DIV', 'MathOperator', lambda d: lambda: rs.DIV(d['close'], d['close2']), lambda d: lambda: c_talib.DIV(d['close'], d['close2']))
reg('MAX', 'MathOperator', lambda d: lambda: rs.MAX(d['close'], 30), lambda d: lambda: c_talib.MAX(d['close'], timeperiod=30))
reg('MIN', 'MathOperator', lambda d: lambda: rs.MIN(d['close'], 30), lambda d: lambda: c_talib.MIN(d['close'], timeperiod=30))
reg('SUM', 'MathOperator', lambda d: lambda: rs.SUM(d['close'], 30), lambda d: lambda: c_talib.SUM(d['close'], timeperiod=30))

# --- Cycle ---
reg('HT_DCPERIOD', 'Cycle', lambda d: lambda: rs.HT_DCPERIOD(d['close']), lambda d: lambda: c_talib.HT_DCPERIOD(d['close']))
reg('HT_DCPHASE', 'Cycle', lambda d: lambda: rs.HT_DCPHASE(d['close']), lambda d: lambda: c_talib.HT_DCPHASE(d['close']))
reg('HT_PHASOR', 'Cycle', lambda d: lambda: rs.HT_PHASOR(d['close']), lambda d: lambda: c_talib.HT_PHASOR(d['close']))
reg('HT_SINE', 'Cycle', lambda d: lambda: rs.HT_SINE(d['close']), lambda d: lambda: c_talib.HT_SINE(d['close']))
reg('HT_TRENDMODE', 'Cycle', lambda d: lambda: rs.HT_TRENDMODE(d['close']), lambda d: lambda: c_talib.HT_TRENDMODE(d['close']))

# --- Pattern (样本) ---
for fn in ['CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDL3BLACKCROWS', 'CDL3WHITESOLDIERS',
           'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLHARAMI', 'CDLDARKCLOUDCOVER', 'CDLPIERCING']:
    reg(fn, 'Pattern', lambda d, f=fn: lambda: getattr(rs, f)(d['open'], d['high'], d['low'], d['close']),
                        lambda d, f=fn: lambda: getattr(c_talib, f)(d['open'], d['high'], d['low'], d['close']))


# ============================================================
# 基准运行
# ============================================================

def bench(func, warmup=3, runs=50):
    """返回平均耗时 (微秒)"""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    # 取中位数
    return times[len(times) // 2]


def main():
    print(f"Running benchmarks: {len(FUNCS)} indicators × {len(SIZES)} sizes...")
    print()

    # results[name] = {size: (rs_us, c_us)}
    results = OrderedDict()

    for name, group, build_rs, build_c in FUNCS:
        results[name] = {'group': group}
        for n in SIZES:
            d = DATA[n]
            try:
                rs_func = build_rs(d)
                c_func = build_c(d)
                rs_us = bench(rs_func)
                c_us = bench(c_func)
                results[name][n] = (rs_us, c_us)
            except Exception as e:
                results[name][n] = (float('nan'), float('nan'))
        # 进度
        sys.stdout.write(f"\r  {name:30s} done")
        sys.stdout.flush()

    print("\n\nGenerating report...")

    # ---- 生成 Markdown ----
    lines = []
    lines.append("# talib-rs vs C TA-Lib 性能对比报告\n")
    lines.append(f"> 自动生成 | 指标数: {len(FUNCS)} | 数据集: 1K / 10K / 100K\n")
    lines.append("")
    lines.append("所有时间单位: **微秒 (μs)**，数值越小越快。")
    lines.append("**Ratio** = C TA-Lib / talib-rs，>1.0 表示 talib-rs 更快。\n")

    # 按 group 分组
    groups = OrderedDict()
    for name, info in results.items():
        g = info['group']
        if g not in groups:
            groups[g] = []
        groups[g].append(name)

    for group, names in groups.items():
        lines.append(f"\n## {group}\n")
        lines.append("| Indicator | 1K rs | 1K C | Ratio | 10K rs | 10K C | Ratio | 100K rs | 100K C | Ratio |")
        lines.append("|-----------|------:|-----:|------:|-------:|------:|------:|--------:|-------:|------:|")

        for name in names:
            info = results[name]
            cells = [f"| {name:s} "]
            for n in SIZES:
                rs_us, c_us = info.get(n, (float('nan'), float('nan')))
                if np.isnan(rs_us):
                    cells.append("| - | - | - ")
                else:
                    ratio = c_us / rs_us if rs_us > 0 else 0
                    marker = "**" if ratio > 1.05 else ""
                    cells.append(f"| {rs_us:.1f} | {c_us:.1f} | {marker}{ratio:.2f}x{marker} ")
            cells.append("|")
            lines.append("".join(cells))

    # 汇总统计
    lines.append("\n## 汇总\n")
    for n in SIZES:
        faster = 0
        slower = 0
        equal = 0
        total_rs = 0
        total_c = 0
        for name, info in results.items():
            rs_us, c_us = info.get(n, (float('nan'), float('nan')))
            if not np.isnan(rs_us):
                total_rs += rs_us
                total_c += c_us
                ratio = c_us / rs_us
                if ratio > 1.05:
                    faster += 1
                elif ratio < 0.95:
                    slower += 1
                else:
                    equal += 1
        lines.append(f"**{n//1000}K 数据集**: talib-rs 更快 {faster} 个 | 持平 {equal} 个 | C 更快 {slower} 个 | "
                     f"总耗时 talib-rs={total_rs:.0f}μs vs C={total_c:.0f}μs\n")

    report = "\n".join(lines) + "\n"

    # 写入文件
    out_path = Path(__file__).parent.parent / "BENCHMARK.md"
    out_path.write_text(report, encoding='utf-8')
    print(f"\nReport written to {out_path}")
    print(report[:2000])


if __name__ == '__main__':
    main()
