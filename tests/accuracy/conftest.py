"""精度验证测试的共享配置和辅助函数"""

import numpy as np
import pytest

# 相对/绝对容差
RTOL = 1e-10
ATOL = 1e-12


def assert_talib_equal(our_result, their_result, rtol=RTOL, atol=ATOL):
    """对比两个输出数组，处理 NaN lookback 区域"""
    np.testing.assert_allclose(
        our_result, their_result,
        rtol=rtol, atol=atol, equal_nan=True
    )


def make_close(n=1000, seed=42):
    """生成模拟收盘价数据"""
    rng = np.random.RandomState(seed)
    # 随机游走模拟价格
    returns = rng.normal(0.0005, 0.02, n)
    price = 100.0 * np.exp(np.cumsum(returns))
    return price


def make_ohlcv(n=1000, seed=42):
    """生成模拟 OHLCV 数据"""
    rng = np.random.RandomState(seed)
    close = make_close(n, seed)

    # 从 close 生成 OHLC
    spread = close * 0.02  # 2% 波动
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.uniform(1e6, 1e7, n)

    return open_, high, low, close, volume
