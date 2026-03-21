"""
talib-rs: Pure Rust technical analysis library — drop-in replacement for TA-Lib.

Usage:
    import talib
    import numpy as np

    close = np.random.random(100)
    sma = talib.SMA(close, timeperiod=20)
    rsi = talib.RSI(close, timeperiod=14)
    macd, signal, hist = talib.MACD(close)
"""

# 从 Rust 扩展导入所有函数
from talib._talib import *  # noqa: F401, F403
from talib._talib import get_functions, get_function_groups

__version__ = "0.1.0"
__ta_version__ = "0.6.4"

# MA_Type 常量（与原版 TA-Lib 兼容）
class MA_Type:
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8
