// K线形态识别 — 61 种形态
// 输出: Vec<i32>，-100 = 空头信号, 0 = 无信号, +100 = 多头信号

use crate::error::{TaError, TaResult};

// ========== 辅助函数 ==========

/// K线辅助函数: 实体大小
#[inline]
fn real_body(open: f64, close: f64) -> f64 {
    (close - open).abs()
}

/// K线辅助函数: 上影线长度
#[inline]
fn upper_shadow(open: f64, high: f64, close: f64) -> f64 {
    high - open.max(close)
}

/// K线辅助函数: 下影线长度
#[inline]
fn lower_shadow(open: f64, low: f64, close: f64) -> f64 {
    open.min(close) - low
}

/// K线辅助函数: 是否为阳线
#[inline]
fn is_bullish(open: f64, close: f64) -> bool {
    close > open
}

/// K线辅助函数: 是否为十字星 (body < range * threshold)
#[inline]
fn is_doji(open: f64, high: f64, low: f64, close: f64, threshold: f64) -> bool {
    let range = high - low;
    range > 0.0 && real_body(open, close) / range < threshold
}

/// K线辅助函数: 近10根K线的平均实体大小
#[inline]
fn body_avg(open: &[f64], close: &[f64], i: usize) -> f64 {
    let lookback = 10.min(i + 1);
    let start = i + 1 - lookback;
    let sum: f64 = (start..=i).map(|j| real_body(open[j], close[j])).sum();
    sum / lookback as f64
}

/// 简单趋势判断: 最近n根K线 close 是否上升
#[inline]
fn is_uptrend(close: &[f64], i: usize, n: usize) -> bool {
    if i < n {
        return false;
    }
    close[i] > close[i - n]
}

/// 简单趋势判断: 最近n根K线 close 是否下降
#[inline]
fn is_downtrend(close: &[f64], i: usize, n: usize) -> bool {
    if i < n {
        return false;
    }
    close[i] < close[i - n]
}

/// 容差相等
#[inline]
fn near_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// 验证 OHLC 数组长度
fn validate_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<usize> {
    let len = open.len();
    if len != high.len() || len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: high.len().min(low.len()).min(close.len()),
        });
    }
    Ok(len)
}

// ========== 已实现的3种基础形态 ==========

/// CDL_DOJI — 十字星
pub fn cdl_doji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];

    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        // 实体非常小（< 范围的 10%）
        if range > 0.0 && body / range < 0.1 {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_HAMMER — 锤子线
pub fn cdl_hammer(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];

    for i in 1..len {
        let body = real_body(open[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let range = high[i] - low[i];

        // 锤子线: 小实体在上方, 长下影线(>=实体2倍), 几乎无上影线
        if range > 0.0 && body > 0.0 && lower >= 2.0 * body && upper <= body * 0.3 {
            output[i] = 100;
        }
    }
    Ok(output)
}

/// CDL_ENGULFING — 吞没形态
pub fn cdl_engulfing(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];

    for i in 1..len {
        let prev_body = real_body(open[i - 1], close[i - 1]);
        let curr_body = real_body(open[i], close[i]);

        if curr_body > prev_body {
            // 多头吞没: 前阴后阳，当前实体包含前一根
            if !is_bullish(open[i - 1], close[i - 1])
                && is_bullish(open[i], close[i])
                && open[i] <= close[i - 1]
                && close[i] >= open[i - 1]
            {
                output[i] = 100;
            }
            // 空头吞没: 前阳后阴
            else if is_bullish(open[i - 1], close[i - 1])
                && !is_bullish(open[i], close[i])
                && open[i] >= close[i - 1]
                && close[i] <= open[i - 1]
            {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

// ========== 单根K线形态 ==========

/// CDL_CLOSINGMARUBOZU — 收盘光头光脚线 (body > 80% of range, tiny shadows)
pub fn cdl_closingmarubozu(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range > 0.0 && body / range > 0.8 {
            let upper = upper_shadow(open[i], high[i], close[i]);
            let lower = lower_shadow(open[i], low[i], close[i]);
            // 收盘端的影线极短
            if is_bullish(open[i], close[i]) && upper <= body * 0.05 {
                // 阳线: 上影线极短（收盘在最高点附近）
                output[i] = 100;
            } else if !is_bullish(open[i], close[i]) && lower <= body * 0.05 {
                // 阴线: 下影线极短（收盘在最低点附近）
                output[i] = -100;
            }
            let _ = (upper, lower); // suppress unused warning
        }
    }
    Ok(output)
}

/// CDL_DRAGONFLYDOJI — 蜻蜓十字 (长下影线，无上影线)
pub fn cdl_dragonflydoji(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        // 十字星 + 长下影线 + 几乎无上影线
        if body / range < 0.1 && lower >= range * 0.6 && upper <= range * 0.1 {
            output[i] = 100;
        }
    }
    Ok(output)
}

/// CDL_GRAVESTONEDOJI — 墓碑十字 (长上影线，无下影线)
pub fn cdl_gravestonedoji(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        if body / range < 0.1 && upper >= range * 0.6 && lower <= range * 0.1 {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_HIGHWAVE — 大浪线 (小实体，两侧长影线)
pub fn cdl_highwave(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let avg = body_avg(open, close, i);
        // 小实体 + 两侧影线都长于实体3倍
        if body < avg * 0.3 && upper > body * 3.0 && lower > body * 3.0 {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_LONGLEGGEDDOJI — 长脚十字 (十字星 + 上下影线都长)
pub fn cdl_longleggeddoji(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        if body / range < 0.1 && upper >= range * 0.3 && lower >= range * 0.3 {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_LONGLINE — 长线 (实体 >= 70% of range)
pub fn cdl_longline(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range > 0.0 && body / range >= 0.7 {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_MARUBOZU — 光头光脚线 (无影线)
pub fn cdl_marubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 || body <= 0.0 {
            continue;
        }
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let tol = range * 0.01; // 1% 容差
        if upper <= tol && lower <= tol {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_RICKSHAWMAN — 黄包车夫 (十字星 + 实体在中间 + 长影线)
pub fn cdl_rickshawman(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let mid = (high[i] + low[i]) / 2.0;
        let body_center = (open[i] + close[i]) / 2.0;
        // 十字星 + 实体在中间 + 两侧影线长
        if body / range < 0.1
            && (body_center - mid).abs() / range < 0.1
            && upper >= range * 0.3
            && lower >= range * 0.3
        {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_SHORTLINE — 短线 (实体 <= 30% of range)
pub fn cdl_shortline(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range > 0.0 && body / range <= 0.3 && body > 0.0 {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_SPINNINGTOP — 纺锤线 (小实体居中, 两侧有影线)
pub fn cdl_spinningtop(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let avg = body_avg(open, close, i);
        // 小实体 + 两侧有影线
        if body < avg * 0.5 && upper > body && lower > body && body > 0.0 {
            output[i] = if is_bullish(open[i], close[i]) {
                100
            } else {
                -100
            };
        }
    }
    Ok(output)
}

/// CDL_TAKURI — 探底线 (类似蜻蜓十字，但实体可稍大)
pub fn cdl_takuri(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 0..len {
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        // 小实体在上方 + 下影线长(>=实体3倍) + 上影线短
        if body / range < 0.2 && lower >= body * 3.0 && upper <= range * 0.1 {
            output[i] = 100;
        }
    }
    Ok(output)
}

// ========== 两根K线形态 ==========

/// CDL_2CROWS — 两只乌鸦 (上升趋势中，跳空高开阴线 + 第二阴线吞没第一阴线)
pub fn cdl_2crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 第一根阳线(上升趋势)
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        let avg = body_avg(open, close, i - 2);
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根阴线跳空高开
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if open[i - 1] <= close[i - 2] {
            continue;
        }
        // 第三根阴线: 开盘在第二根实体内, 收盘在第一根实体内
        if is_bullish(open[i], close[i]) {
            continue;
        }
        if open[i] >= open[i - 1] || open[i] <= close[i - 1] {
            continue;
        }
        if close[i] >= open[i - 2] || close[i] <= close[i - 2] {
            continue;
        }
        output[i] = -100;
    }
    Ok(output)
}

/// CDL_COUNTERATTACK — 反击线 (两根颜色相反的K线，收盘价几乎相同)
pub fn cdl_counterattack(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let avg = body_avg(open, close, i);
        let tol = avg * 0.05;
        let prev_bull = is_bullish(open[i - 1], close[i - 1]);
        let curr_bull = is_bullish(open[i], close[i]);
        // 方向相反 + 收盘价接近 + 实体都较大
        if prev_bull == curr_bull {
            continue;
        }
        if !near_eq(close[i - 1], close[i], tol) {
            continue;
        }
        if real_body(open[i - 1], close[i - 1]) < avg * 0.5 {
            continue;
        }
        if real_body(open[i], close[i]) < avg * 0.5 {
            continue;
        }
        // 多头反击: 前阴后阳
        if !prev_bull && curr_bull {
            output[i] = 100;
        } else {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_DARKCLOUDCOVER — 乌云盖顶 (阳线后阴线，开盘高于前高，收盘低于前实体中点)
pub fn cdl_darkcloudcover(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        if !is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            continue;
        }
        let prev_mid = (open[i - 1] + close[i - 1]) / 2.0;
        // 阴线开盘高于前高, 收盘低于前实体中点但高于前开盘
        if open[i] >= high[i - 1] && close[i] < prev_mid && close[i] > open[i - 1] {
            let avg = body_avg(open, close, i);
            if real_body(open[i - 1], close[i - 1]) > avg * 0.5 {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

/// CDL_DOJISTAR — 十字星 (长实体 + 跳空十字星)
pub fn cdl_dojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let avg = body_avg(open, close, i - 1);
        let prev_body = real_body(open[i - 1], close[i - 1]);
        if prev_body < avg {
            continue;
        }
        if !is_doji(open[i], high[i], low[i], close[i], 0.1) {
            continue;
        }
        // 跳空判断
        if is_bullish(open[i - 1], close[i - 1]) {
            // 阳线后向上跳空
            if low[i] > close[i - 1] {
                output[i] = -100; // 看跌十字星
            }
        } else {
            // 阴线后向下跳空
            if high[i] < close[i - 1] {
                output[i] = 100; // 看涨十字星
            }
        }
    }
    Ok(output)
}

/// CDL_HANGINGMAN — 上吊线 (上升趋势中的锤子线形态，看空)
pub fn cdl_hangingman(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let body = real_body(open[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 || body <= 0.0 {
            continue;
        }
        // 锤子形态 + 上升趋势
        if lower >= 2.0 * body && upper <= body * 0.3 && is_uptrend(close, i - 1, 3) {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_HARAMI — 孕线 (第二根实体完全在第一根实体内)
pub fn cdl_harami(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let prev_body = real_body(open[i - 1], close[i - 1]);
        let curr_body = real_body(open[i], close[i]);
        let avg = body_avg(open, close, i - 1);
        if prev_body < avg * 0.5 || curr_body >= prev_body {
            continue;
        }
        let prev_top = open[i - 1].max(close[i - 1]);
        let prev_bot = open[i - 1].min(close[i - 1]);
        let curr_top = open[i].max(close[i]);
        let curr_bot = open[i].min(close[i]);
        if curr_top <= prev_top && curr_bot >= prev_bot {
            // 多头孕线: 前阴后阳
            if !is_bullish(open[i - 1], close[i - 1]) && is_bullish(open[i], close[i]) {
                output[i] = 100;
            }
            // 空头孕线: 前阳后阴
            else if is_bullish(open[i - 1], close[i - 1]) && !is_bullish(open[i], close[i]) {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

/// CDL_HARAMICROSS — 十字孕线 (孕线 + 第二根为十字星)
pub fn cdl_haramicross(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let prev_body = real_body(open[i - 1], close[i - 1]);
        let avg = body_avg(open, close, i - 1);
        if prev_body < avg * 0.5 {
            continue;
        }
        if !is_doji(open[i], high[i], low[i], close[i], 0.1) {
            continue;
        }
        let prev_top = open[i - 1].max(close[i - 1]);
        let prev_bot = open[i - 1].min(close[i - 1]);
        let curr_top = open[i].max(close[i]);
        let curr_bot = open[i].min(close[i]);
        if curr_top <= prev_top && curr_bot >= prev_bot {
            if !is_bullish(open[i - 1], close[i - 1]) {
                output[i] = 100;
            } else {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

/// CDL_HIKKAKE — 陷阱形态 (内包线 + 突破失败)
pub fn cdl_hikkake(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    // 需要至少5根K线: inside bar at i-1,i, confirmation at i+1..i+3
    if len < 5 {
        return Ok(output);
    }
    for i in 1..len.saturating_sub(3) {
        // 内包线: 当前高低在前一根范围内
        if high[i] >= high[i - 1] || low[i] <= low[i - 1] {
            continue;
        }
        // 检查后续3根是否有确认 (突破失败)
        for j in (i + 1)..=(i + 3).min(len - 1) {
            // 多头陷阱: 向下突破后反转向上
            if close[j] > high[i] {
                output[j] = 100;
                break;
            }
            // 空头陷阱: 向上突破后反转向下
            if close[j] < low[i] {
                output[j] = -100;
                break;
            }
        }
    }
    Ok(output)
}

/// CDL_HIKKAKEMOD — 修正陷阱形态
pub fn cdl_hikkakemod(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    if len < 6 {
        return Ok(output);
    }
    for i in 2..len.saturating_sub(3) {
        // 第一根为长实体
        let avg = body_avg(open, close, i - 2);
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // i-1 和 i 构成内包线
        if high[i] >= high[i - 1] || low[i] <= low[i - 1] {
            continue;
        }
        if high[i - 1] >= high[i - 2] || low[i - 1] <= low[i - 2] {
            continue;
        }
        for j in (i + 1)..=(i + 3).min(len - 1) {
            if close[j] > high[i] {
                output[j] = 100;
                break;
            }
            if close[j] < low[i] {
                output[j] = -100;
                break;
            }
        }
    }
    Ok(output)
}

/// CDL_HOMINGPIGEON — 家鸽 (两根阴线，第二根在第一根内部)
pub fn cdl_homingpigeon(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        // 两根都是阴线
        if is_bullish(open[i - 1], close[i - 1]) || is_bullish(open[i], close[i]) {
            continue;
        }
        let avg = body_avg(open, close, i - 1);
        if real_body(open[i - 1], close[i - 1]) < avg {
            continue;
        }
        // 第二根实体在第一根内
        if close[i] >= close[i - 1] && open[i] <= open[i - 1] {
            output[i] = 100; // 看涨反转
        }
    }
    Ok(output)
}

/// CDL_INNECK — 颈内线 (阴线 + 小阳线收盘接近前低)
pub fn cdl_inneck(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        let avg = body_avg(open, close, i);
        let tol = avg * 0.1;
        // 阳线开盘低于前收盘，收盘接近前低但略高
        if open[i] < close[i - 1] && near_eq(close[i], low[i - 1], tol) && close[i] <= close[i - 1]
        {
            output[i] = -100; // 看跌延续
        }
    }
    Ok(output)
}

/// CDL_INVERTEDHAMMER — 倒锤子线 (下降趋势中，小实体在下方，长上影线)
pub fn cdl_invertedhammer(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 || body <= 0.0 {
            continue;
        }
        // 长上影线(>=实体2倍) + 短下影线 + 下降趋势
        if upper >= 2.0 * body && lower <= body * 0.3 && is_downtrend(close, i - 1, 3) {
            output[i] = 100; // 看涨反转
        }
    }
    Ok(output)
}

/// CDL_KICKING — 反冲形态 (一色光头光脚 + 反色光头光脚 + 跳空)
pub fn cdl_kicking(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let range1 = high[i - 1] - low[i - 1];
        let range2 = high[i] - low[i];
        if range1 <= 0.0 || range2 <= 0.0 {
            continue;
        }
        let body1 = real_body(open[i - 1], close[i - 1]);
        let body2 = real_body(open[i], close[i]);
        let tol1 = range1 * 0.02;
        let tol2 = range2 * 0.02;
        let us1 = upper_shadow(open[i - 1], high[i - 1], close[i - 1]);
        let ls1 = lower_shadow(open[i - 1], low[i - 1], close[i - 1]);
        let us2 = upper_shadow(open[i], high[i], close[i]);
        let ls2 = lower_shadow(open[i], low[i], close[i]);
        // 两根都是光头光脚
        let maru1 = us1 <= tol1 && ls1 <= tol1 && body1 > 0.0;
        let maru2 = us2 <= tol2 && ls2 <= tol2 && body2 > 0.0;
        if !maru1 || !maru2 {
            continue;
        }
        let bull1 = is_bullish(open[i - 1], close[i - 1]);
        let bull2 = is_bullish(open[i], close[i]);
        if bull1 == bull2 {
            continue;
        } // 颜色必须相反
          // 跳空 + 方向
        if !bull1 && bull2 && open[i] > open[i - 1] {
            output[i] = 100; // 多头反冲
        } else if bull1 && !bull2 && open[i] < open[i - 1] {
            output[i] = -100; // 空头反冲
        }
    }
    Ok(output)
}

/// CDL_KICKINGBYLENGTH — 反冲形态(按长度) — 与 kicking 类似，取较长实体方向
pub fn cdl_kickingbylength(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let range1 = high[i - 1] - low[i - 1];
        let range2 = high[i] - low[i];
        if range1 <= 0.0 || range2 <= 0.0 {
            continue;
        }
        let body1 = real_body(open[i - 1], close[i - 1]);
        let body2 = real_body(open[i], close[i]);
        let tol1 = range1 * 0.02;
        let tol2 = range2 * 0.02;
        let us1 = upper_shadow(open[i - 1], high[i - 1], close[i - 1]);
        let ls1 = lower_shadow(open[i - 1], low[i - 1], close[i - 1]);
        let us2 = upper_shadow(open[i], high[i], close[i]);
        let ls2 = lower_shadow(open[i], low[i], close[i]);
        let maru1 = us1 <= tol1 && ls1 <= tol1 && body1 > 0.0;
        let maru2 = us2 <= tol2 && ls2 <= tol2 && body2 > 0.0;
        if !maru1 || !maru2 {
            continue;
        }
        let bull1 = is_bullish(open[i - 1], close[i - 1]);
        let bull2 = is_bullish(open[i], close[i]);
        if bull1 == bull2 {
            continue;
        }
        // 按较长实体的方向决定信号
        if body2 >= body1 {
            output[i] = if bull2 { 100 } else { -100 };
        } else {
            output[i] = if bull1 { 100 } else { -100 };
        }
    }
    Ok(output)
}

/// CDL_MATCHINGLOW — 相同低价 (两根阴线收盘相同)
pub fn cdl_matchinglow(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        if is_bullish(open[i - 1], close[i - 1]) || is_bullish(open[i], close[i]) {
            continue;
        }
        let avg = body_avg(open, close, i);
        let tol = avg * 0.03;
        if near_eq(close[i - 1], close[i], tol) {
            output[i] = 100; // 看涨反转
        }
    }
    Ok(output)
}

/// CDL_ONNECK — 颈上线 (阴线 + 阳线收盘在前低附近)
pub fn cdl_onneck(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        let avg = body_avg(open, close, i);
        let tol = avg * 0.05;
        if open[i] < low[i - 1] && near_eq(close[i], low[i - 1], tol) {
            output[i] = -100; // 看跌延续
        }
    }
    Ok(output)
}

/// CDL_PIERCING — 刺透形态 (阴线后阳线，开盘低于前低，收盘高于前实体中点)
pub fn cdl_piercing(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        let prev_mid = (open[i - 1] + close[i - 1]) / 2.0;
        if open[i] < low[i - 1] && close[i] > prev_mid && close[i] < open[i - 1] {
            let avg = body_avg(open, close, i);
            if real_body(open[i - 1], close[i - 1]) > avg * 0.5 {
                output[i] = 100;
            }
        }
    }
    Ok(output)
}

/// CDL_SEPARATINGLINES — 分离线 (相同开盘价，方向相反)
pub fn cdl_separatinglines(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let avg = body_avg(open, close, i);
        let tol = avg * 0.05;
        let bull1 = is_bullish(open[i - 1], close[i - 1]);
        let bull2 = is_bullish(open[i], close[i]);
        if bull1 == bull2 {
            continue;
        }
        if !near_eq(open[i - 1], open[i], tol) {
            continue;
        }
        if real_body(open[i], close[i]) < avg * 0.5 {
            continue;
        }
        output[i] = if bull2 { 100 } else { -100 };
    }
    Ok(output)
}

/// CDL_SHOOTINGSTAR — 流星线 (上升趋势中倒锤子形态，看空)
pub fn cdl_shootingstar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let body = real_body(open[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 || body <= 0.0 {
            continue;
        }
        // 长上影线 + 短下影线 + 上升趋势
        if upper >= 2.0 * body && lower <= body * 0.3 && is_uptrend(close, i - 1, 3) {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_STICKSANDWICH — 条形三明治 (阴-阳-阴，第一和第三收盘相同)
pub fn cdl_sticksandwich(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if !is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            continue;
        }
        let avg = body_avg(open, close, i);
        let tol = avg * 0.03;
        if near_eq(close[i - 2], close[i], tol) {
            output[i] = 100; // 看涨反转
        }
    }
    Ok(output)
}

/// CDL_THRUSTING — 插入形态 (阴线 + 阳线收盘在前实体中点和前收盘之间)
pub fn cdl_thrusting(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        let prev_mid = (open[i - 1] + close[i - 1]) / 2.0;
        // 开盘低于前低, 收盘在前收盘和前实体中点之间
        if open[i] < low[i - 1] && close[i] > close[i - 1] && close[i] < prev_mid {
            output[i] = -100; // 看跌延续
        }
    }
    Ok(output)
}

// ========== 三根及以上K线形态 ==========

/// CDL_3BLACKCROWS — 三只乌鸦 (三根连续下跌阴线)
pub fn cdl_3blackcrows(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 三根都是阴线
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            continue;
        }
        // 每根收盘都更低
        if close[i - 1] >= close[i - 2] || close[i] >= close[i - 1] {
            continue;
        }
        // 每根开盘在前一根实体内
        if open[i - 1] > open[i - 2] || open[i - 1] < close[i - 2] {
            continue;
        }
        if open[i] > open[i - 1] || open[i] < close[i - 1] {
            continue;
        }
        let avg = body_avg(open, close, i);
        // 实体都较大
        if real_body(open[i], close[i]) > avg * 0.5
            && real_body(open[i - 1], close[i - 1]) > avg * 0.5
            && real_body(open[i - 2], close[i - 2]) > avg * 0.5
        {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_3INSIDE — 三内部上升/下降 (孕线 + 确认K线)
pub fn cdl_3inside(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        let top1 = open[i - 2].max(close[i - 2]);
        let bot1 = open[i - 2].min(close[i - 2]);
        let top2 = open[i - 1].max(close[i - 1]);
        let bot2 = open[i - 1].min(close[i - 1]);
        // 孕线: 第二根在第一根内
        if top2 > top1 || bot2 < bot1 {
            continue;
        }
        let avg = body_avg(open, close, i - 2);
        if real_body(open[i - 2], close[i - 2]) < avg * 0.5 {
            continue;
        }
        // 三内部上升: 前阴+后阳孕线 + 第三根阳线收盘高于第一根开盘
        if !is_bullish(open[i - 2], close[i - 2])
            && is_bullish(open[i - 1], close[i - 1])
            && is_bullish(open[i], close[i])
            && close[i] > top1
        {
            output[i] = 100;
        }
        // 三内部下降: 前阳+后阴孕线 + 第三根阴线收盘低于第一根开盘
        else if is_bullish(open[i - 2], close[i - 2])
            && !is_bullish(open[i - 1], close[i - 1])
            && !is_bullish(open[i], close[i])
            && close[i] < bot1
        {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_3LINESTRIKE — 三线打击 (三同向K线 + 第四根反向吞没)
pub fn cdl_3linestrike(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 3..len {
        // 多头三线打击: 三根阳线 + 第四根阴线吞没
        if is_bullish(open[i - 3], close[i - 3])
            && is_bullish(open[i - 2], close[i - 2])
            && is_bullish(open[i - 1], close[i - 1])
            && close[i - 2] > close[i - 3]
            && close[i - 1] > close[i - 2]
            && !is_bullish(open[i], close[i])
            && open[i] >= close[i - 1]
            && close[i] <= open[i - 3]
        {
            output[i] = 100; // 看涨延续
        }
        // 空头三线打击: 三根阴线 + 第四根阳线吞没
        else if !is_bullish(open[i - 3], close[i - 3])
            && !is_bullish(open[i - 2], close[i - 2])
            && !is_bullish(open[i - 1], close[i - 1])
            && close[i - 2] < close[i - 3]
            && close[i - 1] < close[i - 2]
            && is_bullish(open[i], close[i])
            && open[i] <= close[i - 1]
            && close[i] >= open[i - 3]
        {
            output[i] = -100; // 看跌延续
        }
    }
    Ok(output)
}

/// CDL_3OUTSIDE — 三外部上升/下降 (吞没 + 确认K线)
pub fn cdl_3outside(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 多头吞没 + 阳线确认
        if !is_bullish(open[i - 2], close[i - 2])
            && is_bullish(open[i - 1], close[i - 1])
            && open[i - 1] <= close[i - 2]
            && close[i - 1] >= open[i - 2]
            && real_body(open[i - 1], close[i - 1]) > real_body(open[i - 2], close[i - 2])
            && is_bullish(open[i], close[i])
            && close[i] > close[i - 1]
        {
            output[i] = 100;
        }
        // 空头吞没 + 阴线确认
        else if is_bullish(open[i - 2], close[i - 2])
            && !is_bullish(open[i - 1], close[i - 1])
            && open[i - 1] >= close[i - 2]
            && close[i - 1] <= open[i - 2]
            && real_body(open[i - 1], close[i - 1]) > real_body(open[i - 2], close[i - 2])
            && !is_bullish(open[i], close[i])
            && close[i] < close[i - 1]
        {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_3STARSINSOUTH — 南方三星 (三根递减阴线，影线逐渐缩短)
pub fn cdl_3starsinsouth(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 三根都是阴线
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            continue;
        }
        let b0 = real_body(open[i - 2], close[i - 2]);
        let b1 = real_body(open[i - 1], close[i - 1]);
        let b2 = real_body(open[i], close[i]);
        // 实体递减
        if b1 >= b0 || b2 >= b1 {
            continue;
        }
        // 第一根有长下影线
        let ls0 = lower_shadow(open[i - 2], low[i - 2], close[i - 2]);
        if ls0 < b0 * 0.5 {
            continue;
        }
        // 第二根下影线短于第一根
        let ls1 = lower_shadow(open[i - 1], low[i - 1], close[i - 1]);
        if ls1 >= ls0 {
            continue;
        }
        // 第三根为短实体小范围K线 (Marubozu-like)
        let range2 = high[i] - low[i];
        if range2 <= 0.0 {
            continue;
        }
        if b2 / range2 < 0.5 {
            continue;
        }
        // 低点递增
        if low[i - 1] < low[i - 2] || low[i] < low[i - 1] {
            continue;
        }
        output[i] = 100; // 看涨反转
    }
    Ok(output)
}

/// CDL_3WHITESOLDIERS — 三白兵 (三根连续上涨阳线)
pub fn cdl_3whitesoldiers(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if !is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        // 收盘递增
        if close[i - 1] <= close[i - 2] || close[i] <= close[i - 1] {
            continue;
        }
        // 每根开盘在前一根实体内
        if open[i - 1] < open[i - 2] || open[i - 1] > close[i - 2] {
            continue;
        }
        if open[i] < open[i - 1] || open[i] > close[i - 1] {
            continue;
        }
        let avg = body_avg(open, close, i);
        if real_body(open[i], close[i]) > avg * 0.5
            && real_body(open[i - 1], close[i - 1]) > avg * 0.5
            && real_body(open[i - 2], close[i - 2]) > avg * 0.5
        {
            output[i] = 100;
        }
    }
    Ok(output)
}

/// CDL_ABANDONEDBABY — 弃婴形态 (跳空十字星，两侧均有跳空)
pub fn cdl_abandonedbaby(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 中间是十字星
        if !is_doji(open[i - 1], high[i - 1], low[i - 1], close[i - 1], 0.1) {
            continue;
        }
        let avg = body_avg(open, close, i);
        // 第一根和第三根实体较大
        if real_body(open[i - 2], close[i - 2]) < avg * 0.5 {
            continue;
        }
        if real_body(open[i], close[i]) < avg * 0.5 {
            continue;
        }
        // 多头弃婴: 阴线 + 向下跳空十字星 + 向上跳空阳线
        if !is_bullish(open[i - 2], close[i - 2])
            && high[i - 1] < low[i - 2]
            && low[i] > high[i - 1]
            && is_bullish(open[i], close[i])
        {
            output[i] = 100;
        }
        // 空头弃婴: 阳线 + 向上跳空十字星 + 向下跳空阴线
        else if is_bullish(open[i - 2], close[i - 2])
            && low[i - 1] > high[i - 2]
            && high[i] < low[i - 1]
            && !is_bullish(open[i], close[i])
        {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_ADVANCEBLOCK — 前进受阻 (三根阳线，实体递减，上影线递增)
pub fn cdl_advanceblock(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if !is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        // 收盘递增
        if close[i - 1] <= close[i - 2] || close[i] <= close[i - 1] {
            continue;
        }
        let b0 = real_body(open[i - 2], close[i - 2]);
        let b1 = real_body(open[i - 1], close[i - 1]);
        let b2 = real_body(open[i], close[i]);
        // 实体递减
        if b1 >= b0 || b2 >= b1 {
            continue;
        }
        // 上影线递增
        let us0 = upper_shadow(open[i - 2], high[i - 2], close[i - 2]);
        let us1 = upper_shadow(open[i - 1], high[i - 1], close[i - 1]);
        let us2 = upper_shadow(open[i], high[i], close[i]);
        if us1 <= us0 || us2 <= us1 {
            continue;
        }
        output[i] = -100; // 看跌反转
    }
    Ok(output)
}

/// CDL_BELTHOLD — 捉腰带线 (长开盘光头光脚线，有跳空)
pub fn cdl_belthold(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 1..len {
        let body = real_body(open[i], close[i]);
        let range = high[i] - low[i];
        if range <= 0.0 {
            continue;
        }
        let avg = body_avg(open, close, i);
        if body < avg {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            // 阳线: 开盘即最低点, 跳空低开
            let lower = lower_shadow(open[i], low[i], close[i]);
            if lower <= range * 0.05 && open[i] < low[i - 1] {
                output[i] = 100;
            }
        } else {
            // 阴线: 开盘即最高点, 跳空高开
            let upper = upper_shadow(open[i], high[i], close[i]);
            if upper <= range * 0.05 && open[i] > high[i - 1] {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

/// CDL_BREAKAWAY — 脱离形态 (五根K线突破形态)
pub fn cdl_breakaway(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 4..len {
        let avg = body_avg(open, close, i - 4);
        // 多头脱离: 长阴线 + 跳空向下小K线 + 两根小K线 + 阳线收在第一根实体内
        if !is_bullish(open[i - 4], close[i - 4]) && real_body(open[i - 4], close[i - 4]) > avg {
            if high[i - 3] < close[i - 4] {
                // 向下跳空
                if is_bullish(open[i], close[i])
                    && close[i] > close[i - 4]
                    && close[i] < open[i - 4]
                {
                    output[i] = 100;
                }
            }
        }
        // 空头脱离: 长阳线 + 跳空向上小K线 + 两根小K线 + 阴线收在第一根实体内
        else if is_bullish(open[i - 4], close[i - 4])
            && real_body(open[i - 4], close[i - 4]) > avg
        {
            if low[i - 3] > close[i - 4] {
                // 向上跳空
                if !is_bullish(open[i], close[i])
                    && close[i] < close[i - 4]
                    && close[i] > open[i - 4]
                {
                    output[i] = -100;
                }
            }
        }
    }
    Ok(output)
}

/// CDL_CONCEALBABYSWALL — 藏婴吞没 (四根阴线特殊形态)
pub fn cdl_concealbabyswall(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 3..len {
        // 四根都是阴线
        if is_bullish(open[i - 3], close[i - 3]) {
            continue;
        }
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            continue;
        }
        let range1 = high[i - 3] - low[i - 3];
        let range2 = high[i - 2] - low[i - 2];
        if range1 <= 0.0 || range2 <= 0.0 {
            continue;
        }
        // 前两根为光头光脚阴线 (接近marubozu)
        let us1 = upper_shadow(open[i - 3], high[i - 3], close[i - 3]);
        let ls1 = lower_shadow(open[i - 3], low[i - 3], close[i - 3]);
        let us2 = upper_shadow(open[i - 2], high[i - 2], close[i - 2]);
        let ls2 = lower_shadow(open[i - 2], low[i - 2], close[i - 2]);
        if us1 > range1 * 0.05 || ls1 > range1 * 0.05 {
            continue;
        }
        if us2 > range2 * 0.05 || ls2 > range2 * 0.05 {
            continue;
        }
        // 第三根跳空低开但有长上影线触及第二根实体
        if open[i - 1] >= close[i - 2] {
            continue;
        }
        if high[i - 1] < close[i - 2] {
            continue;
        }
        // 第四根吞没第三根(包括影线)
        if open[i] < high[i - 1] || close[i] > low[i - 1] {
            continue;
        }
        output[i] = 100; // 看涨反转
    }
    Ok(output)
}

/// CDL_EVENINGDOJISTAR — 黄昏十字星 (阳线 + 向上跳空十字星 + 阴线)
pub fn cdl_eveningdojistar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        let avg = body_avg(open, close, i - 2);
        // 第一根长阳线
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根十字星，向上跳空
        if !is_doji(open[i - 1], high[i - 1], low[i - 1], close[i - 1], 0.1) {
            continue;
        }
        if low[i - 1] <= close[i - 2] {
            continue;
        }
        // 第三根阴线，收盘深入第一根实体
        if is_bullish(open[i], close[i]) {
            continue;
        }
        let mid1 = (open[i - 2] + close[i - 2]) / 2.0;
        if close[i] > mid1 {
            continue;
        }
        output[i] = -100;
    }
    Ok(output)
}

/// CDL_EVENINGSTAR — 黄昏星 (阳线 + 向上跳空小实体 + 阴线)
pub fn cdl_eveningstar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        let avg = body_avg(open, close, i - 2);
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根小实体，向上跳空
        if real_body(open[i - 1], close[i - 1]) > avg * 0.3 {
            continue;
        }
        let star_bot = open[i - 1].min(close[i - 1]);
        if star_bot <= close[i - 2] {
            continue;
        }
        // 第三根阴线
        if is_bullish(open[i], close[i]) {
            continue;
        }
        if real_body(open[i], close[i]) < avg * 0.5 {
            continue;
        }
        let mid1 = (open[i - 2] + close[i - 2]) / 2.0;
        if close[i] > mid1 {
            continue;
        }
        output[i] = -100;
    }
    Ok(output)
}

/// CDL_GAPSIDESIDEWHITE — 跳空并列阳线 (跳空后两根相似阳线)
pub fn cdl_gapsidesidewhite(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if !is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        let b1 = real_body(open[i - 1], close[i - 1]);
        let b2 = real_body(open[i], close[i]);
        let avg = body_avg(open, close, i);
        let tol = avg * 0.3;
        // 两根阳线大小相似
        if (b1 - b2).abs() > tol {
            continue;
        }
        // 开盘价接近
        if (open[i - 1] - open[i]).abs() > tol {
            continue;
        }
        // 向上跳空
        if low[i - 1] > high[i - 2] {
            output[i] = 100;
        }
        // 向下跳空
        else if high[i - 1] < low[i - 2] {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_IDENTICAL3CROWS — 相同三鸦 (三阴线，每根开盘等于前一根收盘)
pub fn cdl_identical3crows(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if is_bullish(open[i], close[i]) {
            continue;
        }
        if close[i - 1] >= close[i - 2] || close[i] >= close[i - 1] {
            continue;
        }
        let avg = body_avg(open, close, i);
        let tol = avg * 0.05;
        // 每根开盘等于前一根收盘
        if !near_eq(open[i - 1], close[i - 2], tol) {
            continue;
        }
        if !near_eq(open[i], close[i - 1], tol) {
            continue;
        }
        if real_body(open[i], close[i]) > avg * 0.3 {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_LADDERBOTTOM — 梯底 (五根K线底部反转)
pub fn cdl_ladderbottom(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 4..len {
        // 前三根为阴线，收盘递降
        if is_bullish(open[i - 4], close[i - 4]) {
            continue;
        }
        if is_bullish(open[i - 3], close[i - 3]) {
            continue;
        }
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if close[i - 3] >= close[i - 4] || close[i - 2] >= close[i - 3] {
            continue;
        }
        // 第四根阴线有长上影线 (上影线 > 实体)
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        let us = upper_shadow(open[i - 1], high[i - 1], close[i - 1]);
        if us < real_body(open[i - 1], close[i - 1]) {
            continue;
        }
        // 第五根阳线，跳空高开
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        if open[i] <= open[i - 1] {
            continue;
        }
        output[i] = 100;
    }
    Ok(output)
}

/// CDL_MATHOLD — 铺垫形态 (五根K线延续形态)
pub fn cdl_mathold(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 4..len {
        let avg = body_avg(open, close, i - 4);
        // 多头: 长阳线 + 三根小回调K线(不跌破第一根开盘) + 阳线创新高
        if is_bullish(open[i - 4], close[i - 4]) && real_body(open[i - 4], close[i - 4]) > avg {
            // 中间三根为小实体，低点不低于第一根开盘
            let hold_low = open[i - 4];
            if low[i - 3] < hold_low || low[i - 2] < hold_low || low[i - 1] < hold_low {
                continue;
            }
            let small = avg * 0.5;
            if real_body(open[i - 3], close[i - 3]) > small {
                continue;
            }
            if real_body(open[i - 2], close[i - 2]) > small {
                continue;
            }
            if real_body(open[i - 1], close[i - 1]) > small {
                continue;
            }
            // 第五根阳线创新高
            if is_bullish(open[i], close[i]) && close[i] > close[i - 4] {
                output[i] = 100;
            }
        }
        // 空头类似但方向相反
        else if !is_bullish(open[i - 4], close[i - 4])
            && real_body(open[i - 4], close[i - 4]) > avg
        {
            let hold_high = open[i - 4];
            if high[i - 3] > hold_high || high[i - 2] > hold_high || high[i - 1] > hold_high {
                continue;
            }
            let small = avg * 0.5;
            if real_body(open[i - 3], close[i - 3]) > small {
                continue;
            }
            if real_body(open[i - 2], close[i - 2]) > small {
                continue;
            }
            if real_body(open[i - 1], close[i - 1]) > small {
                continue;
            }
            if !is_bullish(open[i], close[i]) && close[i] < close[i - 4] {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

/// CDL_MORNINGDOJISTAR — 早晨十字星 (阴线 + 向下跳空十字星 + 阳线)
pub fn cdl_morningdojistar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        let avg = body_avg(open, close, i - 2);
        // 第一根长阴线
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根十字星，向下跳空
        if !is_doji(open[i - 1], high[i - 1], low[i - 1], close[i - 1], 0.1) {
            continue;
        }
        if high[i - 1] >= close[i - 2] {
            continue;
        }
        // 第三根阳线，收盘深入第一根实体
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        let mid1 = (open[i - 2] + close[i - 2]) / 2.0;
        if close[i] < mid1 {
            continue;
        }
        output[i] = 100;
    }
    Ok(output)
}

/// CDL_MORNINGSTAR — 早晨星 (阴线 + 向下跳空小实体 + 阳线)
pub fn cdl_morningstar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        let avg = body_avg(open, close, i - 2);
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根小实体，向下跳空
        if real_body(open[i - 1], close[i - 1]) > avg * 0.3 {
            continue;
        }
        let star_top = open[i - 1].max(close[i - 1]);
        if star_top >= close[i - 2] {
            continue;
        }
        // 第三根阳线
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        if real_body(open[i], close[i]) < avg * 0.5 {
            continue;
        }
        let mid1 = (open[i - 2] + close[i - 2]) / 2.0;
        if close[i] < mid1 {
            continue;
        }
        output[i] = 100;
    }
    Ok(output)
}

/// CDL_RISEFALL3METHODS — 上升/下降三法 (五根K线延续形态)
pub fn cdl_risefall3methods(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 4..len {
        let avg = body_avg(open, close, i - 4);
        // 上升三法: 长阳 + 三根小阴线(不跌破第一根低点) + 阳线收盘新高
        if is_bullish(open[i - 4], close[i - 4]) && real_body(open[i - 4], close[i - 4]) > avg {
            let first_low = low[i - 4];
            let first_high = high[i - 4];
            let mid_ok = (1..=3).all(|k| {
                let j = i - 4 + k;
                real_body(open[j], close[j]) < avg * 0.5
                    && low[j] >= first_low
                    && high[j] <= first_high
            });
            if mid_ok && is_bullish(open[i], close[i]) && close[i] > close[i - 4] {
                output[i] = 100;
            }
        }
        // 下降三法: 长阴 + 三根小阳线(不突破第一根高点) + 阴线收盘新低
        else if !is_bullish(open[i - 4], close[i - 4])
            && real_body(open[i - 4], close[i - 4]) > avg
        {
            let first_low = low[i - 4];
            let first_high = high[i - 4];
            let mid_ok = (1..=3).all(|k| {
                let j = i - 4 + k;
                real_body(open[j], close[j]) < avg * 0.5
                    && low[j] >= first_low
                    && high[j] <= first_high
            });
            if mid_ok && !is_bullish(open[i], close[i]) && close[i] < close[i - 4] {
                output[i] = -100;
            }
        }
    }
    Ok(output)
}

/// CDL_STALLEDPATTERN — 停顿形态 (三根阳线，第三根实体缩小)
pub fn cdl_stalledpattern(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        if !is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        if close[i - 1] <= close[i - 2] || close[i] <= close[i - 1] {
            continue;
        }
        let b0 = real_body(open[i - 2], close[i - 2]);
        let b1 = real_body(open[i - 1], close[i - 1]);
        let b2 = real_body(open[i], close[i]);
        let avg = body_avg(open, close, i);
        // 前两根实体大，第三根明显缩小
        if b0 > avg * 0.7 && b1 > avg * 0.7 && b2 < b1 * 0.5 {
            output[i] = -100; // 看跌反转信号
        }
    }
    Ok(output)
}

/// CDL_TASUKIGAP — 跳空并列 (跳空 + 两根部分填补跳空的K线)
pub fn cdl_tasukigap(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 向上跳空: 前两根阳线, 第三根阴线部分填补跳空
        if is_bullish(open[i - 2], close[i - 2]) && is_bullish(open[i - 1], close[i - 1]) {
            if low[i - 1] > high[i - 2] {
                // 向上跳空
                if !is_bullish(open[i], close[i])
                    && open[i] >= open[i - 1]
                    && open[i] <= close[i - 1]
                    && close[i] < open[i - 1]
                    && close[i] > high[i - 2]
                {
                    output[i] = 100; // 看涨延续
                }
            }
        }
        // 向下跳空: 前两根阴线, 第三根阳线部分填补跳空
        if !is_bullish(open[i - 2], close[i - 2]) && !is_bullish(open[i - 1], close[i - 1]) {
            if high[i - 1] < low[i - 2] {
                // 向下跳空
                if is_bullish(open[i], close[i])
                    && open[i] <= open[i - 1]
                    && open[i] >= close[i - 1]
                    && close[i] > open[i - 1]
                    && close[i] < low[i - 2]
                {
                    output[i] = -100; // 看跌延续
                }
            }
        }
    }
    Ok(output)
}

/// CDL_TRISTAR — 三星形态 (三根十字星)
pub fn cdl_tristar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        if !is_doji(open[i - 2], high[i - 2], low[i - 2], close[i - 2], 0.1) {
            continue;
        }
        if !is_doji(open[i - 1], high[i - 1], low[i - 1], close[i - 1], 0.1) {
            continue;
        }
        if !is_doji(open[i], high[i], low[i], close[i], 0.1) {
            continue;
        }
        let mid1 = (open[i - 2] + close[i - 2]) / 2.0;
        let mid2 = (open[i - 1] + close[i - 1]) / 2.0;
        let mid3 = (open[i] + close[i]) / 2.0;
        // 多头三星: 中间十字星向下跳空
        if mid2 < mid1 && mid2 < mid3 {
            output[i] = 100;
        }
        // 空头三星: 中间十字星向上跳空
        else if mid2 > mid1 && mid2 > mid3 {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_UNIQUE3RIVER — 奇特三河底 (三根K线看涨反转)
pub fn cdl_unique3river(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 第一根长阴线
        if is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        let avg = body_avg(open, close, i - 2);
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根阴线孕线 + 有长下影线 + 创新低
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        let top2 = open[i - 1];
        let bot2 = close[i - 1];
        if top2 > open[i - 2] || bot2 < close[i - 2] {
            continue;
        }
        if low[i - 1] >= low[i - 2] {
            continue;
        } // 需要创新低
        let ls = lower_shadow(open[i - 1], low[i - 1], close[i - 1]);
        if ls < real_body(open[i - 1], close[i - 1]) {
            continue;
        }
        // 第三根小阳线，收盘不高于第二根收盘
        if !is_bullish(open[i], close[i]) {
            continue;
        }
        if close[i] > bot2 {
            continue;
        }
        output[i] = 100;
    }
    Ok(output)
}

/// CDL_UPSIDEGAP2CROWS — 向上跳空两只乌鸦
pub fn cdl_upsidegap2crows(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 第一根长阳线
        if !is_bullish(open[i - 2], close[i - 2]) {
            continue;
        }
        let avg = body_avg(open, close, i - 2);
        if real_body(open[i - 2], close[i - 2]) < avg {
            continue;
        }
        // 第二根小阴线，向上跳空
        if is_bullish(open[i - 1], close[i - 1]) {
            continue;
        }
        if close[i - 1] <= close[i - 2] {
            continue;
        } // 跳空
          // 第三根阴线吞没第二根，收盘仍高于第一根收盘
        if is_bullish(open[i], close[i]) {
            continue;
        }
        if open[i] <= open[i - 1] || close[i] >= close[i - 1] {
            continue;
        }
        if close[i] <= close[i - 2] {
            continue;
        }
        output[i] = -100;
    }
    Ok(output)
}

/// CDL_XSIDEGAP3METHODS — 跳空三法 (跳空 + 三根K线确认)
pub fn cdl_xsidegap3methods(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    for i in 2..len {
        // 向上跳空三法: 两根阳线跳空 + 第三根阴线填补跳空
        if is_bullish(open[i - 2], close[i - 2]) && is_bullish(open[i - 1], close[i - 1]) {
            if open[i - 1] > close[i - 2] {
                // 向上跳空
                if !is_bullish(open[i], close[i])
                    && open[i] >= open[i - 1]
                    && open[i] <= close[i - 1]
                    && close[i] >= open[i - 2]
                    && close[i] <= close[i - 2]
                {
                    output[i] = 100; // 看涨延续
                }
            }
        }
        // 向下跳空三法: 两根阴线跳空 + 第三根阳线填补跳空
        if !is_bullish(open[i - 2], close[i - 2]) && !is_bullish(open[i - 1], close[i - 1]) {
            if open[i - 1] < close[i - 2] {
                // 向下跳空
                if is_bullish(open[i], close[i])
                    && open[i] <= open[i - 1]
                    && open[i] >= close[i - 1]
                    && close[i] <= open[i - 2]
                    && close[i] >= close[i - 2]
                {
                    output[i] = -100; // 看跌延续
                }
            }
        }
    }
    Ok(output)
}
