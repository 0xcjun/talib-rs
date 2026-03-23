// Candlestick pattern recognition — 61 patterns
// Exact reimplementation of C TA-Lib's CandleAverage system
// Output: Vec<i32>, -100 = bearish, 0 = no signal, +100 = bullish

use crate::error::{TaError, TaResult};

// ========== C TA-Lib Candle Setting System ==========

/// Range types used by candle settings
#[derive(Clone, Copy)]
enum RangeType {
    RealBody,
    HighLow,
    Shadows,
}

/// A candle setting definition matching C TA-Lib defaults
#[derive(Clone, Copy)]
struct CandleSetting {
    range_type: RangeType,
    avg_period: usize,
    factor: f64,
}

// Default candle settings exactly matching C TA-Lib ta_common.c
const BODY_LONG: CandleSetting = CandleSetting { range_type: RangeType::RealBody, avg_period: 10, factor: 1.0 };
const BODY_VERY_LONG: CandleSetting = CandleSetting { range_type: RangeType::RealBody, avg_period: 10, factor: 3.0 };
const BODY_SHORT: CandleSetting = CandleSetting { range_type: RangeType::RealBody, avg_period: 10, factor: 1.0 };
const BODY_DOJI: CandleSetting = CandleSetting { range_type: RangeType::HighLow, avg_period: 10, factor: 0.1 };
const SHADOW_LONG: CandleSetting = CandleSetting { range_type: RangeType::RealBody, avg_period: 0, factor: 1.0 };
const SHADOW_VERY_LONG: CandleSetting = CandleSetting { range_type: RangeType::RealBody, avg_period: 0, factor: 2.0 };
const SHADOW_SHORT: CandleSetting = CandleSetting { range_type: RangeType::Shadows, avg_period: 10, factor: 1.0 };
const SHADOW_VERY_SHORT: CandleSetting = CandleSetting { range_type: RangeType::HighLow, avg_period: 10, factor: 0.1 };
const NEAR: CandleSetting = CandleSetting { range_type: RangeType::HighLow, avg_period: 5, factor: 0.2 };
const FAR: CandleSetting = CandleSetting { range_type: RangeType::HighLow, avg_period: 5, factor: 0.6 };
const EQUAL: CandleSetting = CandleSetting { range_type: RangeType::HighLow, avg_period: 5, factor: 0.05 };

// ========== Helper Functions ==========

#[inline(always)]
fn real_body(open: f64, close: f64) -> f64 {
    (close - open).abs()
}

#[inline(always)]
fn upper_shadow(open: f64, high: f64, close: f64) -> f64 {
    high - open.max(close)
}

#[inline(always)]
fn lower_shadow(open: f64, low: f64, close: f64) -> f64 {
    open.min(close) - low
}

#[inline(always)]
fn candle_color(open: f64, close: f64) -> i32 {
    if close >= open { 1 } else { -1 }
}

/// Compute the range value for a single bar based on the setting's range_type
#[inline(always)]
fn candle_range(setting: CandleSetting, open: f64, high: f64, low: f64, close: f64) -> f64 {
    match setting.range_type {
        RangeType::RealBody => (close - open).abs(),
        RangeType::HighLow => high - low,
        RangeType::Shadows => {
            (high - low) - (close - open).abs()
        }
    }
}

/// Compute candle average = factor * (sum / avg_period) / divisor
/// When avg_period == 0, use the current bar's range value directly (no averaging)
/// NOTE: C TA-Lib divides by 2.0 when range_type is Shadows
#[inline(always)]
fn candle_average(setting: CandleSetting, sum: f64, open: f64, high: f64, low: f64, close: f64) -> f64 {
    let divisor = match setting.range_type {
        RangeType::Shadows => 2.0,
        _ => 1.0,
    };
    if setting.avg_period > 0 {
        setting.factor * (sum / setting.avg_period as f64) / divisor
    } else {
        setting.factor * candle_range(setting, open, high, low, close) / divisor
    }
}

/// Helper: compute range at index i (unchecked for hot loops)
#[inline(always)]
fn cr(setting: CandleSetting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    unsafe {
        candle_range(setting, *o.get_unchecked(i), *h.get_unchecked(i), *l.get_unchecked(i), *c.get_unchecked(i))
    }
}

/// Helper: compute average at index i with given sum (unchecked for hot loops)
#[inline(always)]
fn ca(setting: CandleSetting, sum: f64, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    unsafe {
        candle_average(setting, sum, *o.get_unchecked(i), *h.get_unchecked(i), *l.get_unchecked(i), *c.get_unchecked(i))
    }
}

/// Helper: real body gap up (min(o,c) of bar2 > max(o,c) of bar1)
#[inline]
fn real_body_gap_up(o: &[f64], c: &[f64], bar2: usize, bar1: usize) -> bool {
    o[bar2].min(c[bar2]) > o[bar1].max(c[bar1])
}

/// Helper: real body gap down (max(o,c) of bar2 < min(o,c) of bar1)
#[inline]
fn real_body_gap_down(o: &[f64], c: &[f64], bar2: usize, bar1: usize) -> bool {
    o[bar2].max(c[bar2]) < o[bar1].min(c[bar1])
}

/// Validate OHLC arrays have same length
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

// ========== Pattern Functions ==========

/// CDL_DOJI
pub fn cdl_doji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let setting = BODY_DOJI;
    let lookback = setting.avg_period;
    if len <= lookback { return Ok(output); }

    // Initialize running sum
    let mut sum = 0.0;
    for i in 0..lookback {
        sum += cr(setting, open, high, low, close, i);
    }

    // BODY_DOJI uses HighLow range type, so inline the specific logic
    // candle_average for HighLow with avg_period>0: factor * (sum / avg_period) / 1.0
    let factor = setting.factor;
    let avg_period_f = setting.avg_period as f64;

    for i in lookback..len {
        unsafe {
            let o_i = *open.get_unchecked(i);
            let h_i = *high.get_unchecked(i);
            let l_i = *low.get_unchecked(i);
            let c_i = *close.get_unchecked(i);

            // BODY_DOJI: threshold = factor * (sum / avg_period) for HighLow range
            let threshold = factor * (sum / avg_period_f);
            if (c_i - o_i).abs() <= threshold {
                *output.get_unchecked_mut(i) = 100;
            }

            // Update running sum: HighLow range = high - low
            let add = h_i - l_i;
            let sub_idx = i - lookback;
            let sub = *high.get_unchecked(sub_idx) - *low.get_unchecked(sub_idx);
            sum += add - sub;
        }
    }
    Ok(output)
}

/// CDL_HAMMER
pub fn cdl_hammer(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_SHORT.avg_period, SHADOW_LONG.avg_period, SHADOW_VERY_SHORT.avg_period, NEAR.avg_period].iter().max().unwrap() + 1;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_long_sum = 0.0;
    let mut shadow_vs_sum = 0.0;
    let mut near_sum = 0.0;

    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }
    // ShadowLong has avg_period=0, no init needed
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - 1 - NEAR.avg_period)..(start - 1) { near_sum += cr(NEAR, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_LONG, shadow_long_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum, open, high, low, close, i)
            && open[i].min(close[i]) <= low[i-1] + ca(NEAR, near_sum, open, high, low, close, i-1)
        {
            output[i] = 100;
        }
        // Update sums
        if BODY_SHORT.avg_period > 0 { body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period); }
        if SHADOW_VERY_SHORT.avg_period > 0 { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period); }
        if NEAR.avg_period > 0 { near_sum += cr(NEAR, open, high, low, close, i-1) - cr(NEAR, open, high, low, close, i - 1 - NEAR.avg_period); }
    }
    Ok(output)
}

/// CDL_ENGULFING
pub fn cdl_engulfing(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    // Engulfing has no candle settings, lookback = 2 (need i-1)
    if len < 2 { return Ok(output); }

    for i in 1..len {
        // Bullish: prev black, curr white, curr close >= prev open, curr open <= prev close
        if candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == 1
            && close[i] >= open[i-1]
            && open[i] <= close[i-1]
        {
            output[i] = 100;
        }
        // Bearish: prev white, curr black, curr open >= prev close, curr close <= prev open
        else if candle_color(open[i-1], close[i-1]) == 1
            && candle_color(open[i], close[i]) == -1
            && open[i] >= close[i-1]
            && close[i] <= open[i-1]
        {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_CLOSINGMARUBOZU
pub fn cdl_closingmarubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_LONG.avg_period)..start { body_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) > ca(BODY_LONG, body_sum, open, high, low, close, i) {
            if candle_color(open[i], close[i]) == 1
                && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            {
                output[i] = 100;
            } else if candle_color(open[i], close[i]) == -1
                && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            {
                output[i] = -100;
            }
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_DRAGONFLYDOJI
pub fn cdl_dragonflydoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_DOJI.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_DOJI.avg_period)..start { body_sum += cr(BODY_DOJI, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
        {
            output[i] = 100;
        }
        body_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_GRAVESTONEDOJI
pub fn cdl_gravestonedoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_DOJI.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_DOJI.avg_period)..start { body_sum += cr(BODY_DOJI, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
        {
            output[i] = 100;
        }
        body_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_HIGHWAVE
pub fn cdl_highwave(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(SHADOW_VERY_LONG.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }
    // SHADOW_VERY_LONG avg_period=0, no init needed

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_VERY_LONG, shadow_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_VERY_LONG, shadow_sum, open, high, low, close, i)
        {
            output[i] = candle_color(open[i], close[i]) * 100;
        }
        if BODY_SHORT.avg_period > 0 { body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period); }
    }
    Ok(output)
}

/// CDL_LONGLEGGEDDOJI
pub fn cdl_longleggeddoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_DOJI.avg_period.max(SHADOW_LONG.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_DOJI.avg_period)..start { body_sum += cr(BODY_DOJI, open, high, low, close, i); }
    // SHADOW_LONG avg_period=0, no init

    for i in start..len {
        if real_body(open[i], close[i]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i)
            && (lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_LONG, shadow_sum, open, high, low, close, i)
                || upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_LONG, shadow_sum, open, high, low, close, i))
        {
            output[i] = 100;
        }
        body_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
    }
    Ok(output)
}

/// CDL_LONGLINE
pub fn cdl_longline(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period.max(SHADOW_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_LONG.avg_period)..start { body_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - SHADOW_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) > ca(BODY_LONG, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_SHORT, shadow_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_SHORT, shadow_sum, open, high, low, close, i)
        {
            output[i] = candle_color(open[i], close[i]) * 100;
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
        shadow_sum += cr(SHADOW_SHORT, open, high, low, close, i) - cr(SHADOW_SHORT, open, high, low, close, i - SHADOW_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_MARUBOZU
pub fn cdl_marubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_LONG.avg_period)..start { body_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) > ca(BODY_LONG, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
        {
            output[i] = candle_color(open[i], close[i]) * 100;
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_RICKSHAWMAN
pub fn cdl_rickshawman(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_DOJI.avg_period, SHADOW_LONG.avg_period, NEAR.avg_period].iter().max().unwrap();
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let mut near_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_DOJI.avg_period)..start { body_sum += cr(BODY_DOJI, open, high, low, close, i); }
    // SHADOW_LONG avg_period=0
    for i in (start - NEAR.avg_period)..start { near_sum += cr(NEAR, open, high, low, close, i); }

    for i in start..len {
        let mid = low[i] + (high[i] - low[i]) / 2.0;
        let near_avg = ca(NEAR, near_sum, open, high, low, close, i);
        if real_body(open[i], close[i]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_LONG, shadow_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_LONG, shadow_sum, open, high, low, close, i)
            && open[i].min(close[i]) <= mid + near_avg
            && open[i].max(close[i]) >= mid - near_avg
        {
            output[i] = 100;
        }
        body_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
        if NEAR.avg_period > 0 { near_sum += cr(NEAR, open, high, low, close, i) - cr(NEAR, open, high, low, close, i - NEAR.avg_period); }
    }
    Ok(output)
}

/// CDL_SHORTLINE
pub fn cdl_shortline(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(SHADOW_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_SHORT, shadow_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_SHORT, shadow_sum, open, high, low, close, i)
        {
            output[i] = candle_color(open[i], close[i]) * 100;
        }
        body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
        shadow_sum += cr(SHADOW_SHORT, open, high, low, close, i) - cr(SHADOW_SHORT, open, high, low, close, i - SHADOW_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_SPINNINGTOP
pub fn cdl_spinningtop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) > real_body(open[i], close[i])
            && lower_shadow(open[i], low[i], close[i]) > real_body(open[i], close[i])
        {
            output[i] = candle_color(open[i], close[i]) * 100;
        }
        body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_TAKURI
pub fn cdl_takuri(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_DOJI.avg_period, SHADOW_VERY_SHORT.avg_period, SHADOW_VERY_LONG.avg_period].iter().max().unwrap();
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_vs_sum = 0.0;
    let mut shadow_vl_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_DOJI.avg_period)..start { body_sum += cr(BODY_DOJI, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    // SHADOW_VERY_LONG avg_period=0

    for i in start..len {
        if real_body(open[i], close[i]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_VERY_LONG, shadow_vl_sum, open, high, low, close, i)
        {
            output[i] = 100;
        }
        body_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
        shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

// ========== Two-candle patterns ==========

/// CDL_2CROWS
pub fn cdl_2crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period + 2;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        // 1st: long white
        if candle_color(open[i-2], close[i-2]) == 1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_sum, open, high, low, close, i-2)
            // 2nd: black, gap up
            && candle_color(open[i-1], close[i-1]) == -1
            && real_body_gap_up(open, close, i-1, i-2)
            // 3rd: black, opens within 2nd body, closes within 1st body
            && candle_color(open[i], close[i]) == -1
            && open[i] < open[i-1] && open[i] > close[i-1]
            && close[i] > open[i-2] && close[i] < close[i-2]
        {
            output[i] = -100;
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_COUNTERATTACK
pub fn cdl_counterattack(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut equal_sum = 0.0;
    let mut body_sum = [0.0f64; 2]; // [0]=current, [1]=prev
    let start = lookback;
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum[1] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_LONG.avg_period)..start { body_sum[0] += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) != candle_color(open[i], close[i])
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum[1], open, high, low, close, i-1)
            && real_body(open[i], close[i]) > ca(BODY_LONG, body_sum[0], open, high, low, close, i)
            && (close[i] - close[i-1]).abs() <= ca(EQUAL, equal_sum, open, high, low, close, i-1)
        {
            output[i] = candle_color(open[i], close[i]) * 100;
        }
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
        body_sum[1] += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_sum[0] += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_DARKCLOUDCOVER
pub fn cdl_darkcloudcover(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.5;
    let lookback = BODY_LONG.avg_period + 1;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == 1
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum, open, high, low, close, i-1)
            && candle_color(open[i], close[i]) == -1
            && open[i] > high[i-1]
            && close[i] > open[i-1]
            && close[i] < close[i-1] - real_body(open[i-1], close[i-1]) * penetration
        {
            output[i] = -100;
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_DOJISTAR
pub fn cdl_dojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_DOJI.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_doji_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_DOJI.avg_period)..start { body_doji_sum += cr(BODY_DOJI, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) <= ca(BODY_DOJI, body_doji_sum, open, high, low, close, i)
        {
            if candle_color(open[i-1], close[i-1]) == 1 && real_body_gap_up(open, close, i, i-1) {
                output[i] = -100;
            } else if candle_color(open[i-1], close[i-1]) == -1 && real_body_gap_down(open, close, i, i-1) {
                output[i] = 100;
            }
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_doji_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
    }
    Ok(output)
}

/// CDL_HANGINGMAN
pub fn cdl_hangingman(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_SHORT.avg_period, SHADOW_LONG.avg_period, SHADOW_VERY_SHORT.avg_period, NEAR.avg_period].iter().max().unwrap() + 1;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_long_sum = 0.0;
    let mut shadow_vs_sum = 0.0;
    let mut near_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - 1 - NEAR.avg_period)..(start - 1) { near_sum += cr(NEAR, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) > ca(SHADOW_LONG, shadow_long_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum, open, high, low, close, i)
            && open[i].min(close[i]) >= high[i-1] - ca(NEAR, near_sum, open, high, low, close, i-1)
        {
            output[i] = -100;
        }
        if BODY_SHORT.avg_period > 0 { body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period); }
        if SHADOW_VERY_SHORT.avg_period > 0 { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period); }
        if NEAR.avg_period > 0 { near_sum += cr(NEAR, open, high, low, close, i-1) - cr(NEAR, open, high, low, close, i - 1 - NEAR.avg_period); }
    }
    Ok(output)
}

/// CDL_HARAMI
pub fn cdl_harami(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) <= ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && open[i].max(close[i]) < open[i-1].max(close[i-1])
            && open[i].min(close[i]) > open[i-1].min(close[i-1])
        {
            output[i] = -candle_color(open[i-1], close[i-1]) * 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_HARAMICROSS
pub fn cdl_haramicross(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_DOJI.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_doji_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_DOJI.avg_period)..start { body_doji_sum += cr(BODY_DOJI, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) <= ca(BODY_DOJI, body_doji_sum, open, high, low, close, i)
            && open[i].max(close[i]) < open[i-1].max(close[i-1])
            && open[i].min(close[i]) > open[i-1].min(close[i-1])
        {
            output[i] = -candle_color(open[i-1], close[i-1]) * 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_doji_sum += cr(BODY_DOJI, open, high, low, close, i) - cr(BODY_DOJI, open, high, low, close, i - BODY_DOJI.avg_period);
    }
    Ok(output)
}

/// CDL_HIKKAKE
pub fn cdl_hikkake(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = 5;
    if len <= lookback { return Ok(output); }

    let mut pattern_idx: i32 = -1;
    let mut pattern_result: i32 = 0;

    // Pre-scan bars before start
    let start = lookback;
    for i in (start.saturating_sub(3))..start {
        if i >= 2 {
            // Inside bar: 2nd has lower high and higher low than 1st
            if high[i-1] < high[i-2] && low[i-1] > low[i-2] {
                // 3rd bar determines direction
                if high[i] < high[i-1] && low[i] < low[i-1] {
                    pattern_result = 100; // bullish
                    pattern_idx = i as i32;
                } else if high[i] > high[i-1] && low[i] > low[i-1] {
                    pattern_result = -100; // bearish
                    pattern_idx = i as i32;
                }
            }
        }
    }

    for i in start..len {
        if i >= 2 && high[i-1] < high[i-2] && low[i-1] > low[i-2] {
            // Inside bar found at i-1,i-2
            if high[i] < high[i-1] && low[i] < low[i-1] {
                pattern_result = 100;
                pattern_idx = i as i32;
                output[i] = pattern_result;
            } else if high[i] > high[i-1] && low[i] > low[i-1] {
                pattern_result = -100;
                pattern_idx = i as i32;
                output[i] = pattern_result;
            } else {
                // Check confirmation
                if pattern_idx >= 0 && (i as i32 - pattern_idx) <= 3 {
                    if pattern_result > 0 && close[i] > high[pattern_idx as usize - 1] {
                        output[i] = pattern_result + 100;
                        pattern_idx = -1;
                    } else if pattern_result < 0 && close[i] < low[pattern_idx as usize - 1] {
                        output[i] = pattern_result - 100;
                        pattern_idx = -1;
                    }
                }
            }
        } else {
            // Check confirmation
            if pattern_idx >= 0 && (i as i32 - pattern_idx) <= 3 {
                if pattern_result > 0 && close[i] > high[pattern_idx as usize - 1] {
                    output[i] = pattern_result + 100;
                    pattern_idx = -1;
                } else if pattern_result < 0 && close[i] < low[pattern_idx as usize - 1] {
                    output[i] = pattern_result - 100;
                    pattern_idx = -1;
                }
            }
        }
    }
    Ok(output)
}

/// CDL_HIKKAKEMOD
pub fn cdl_hikkakemod(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    // C TA-Lib: lookback = max(1, TA_CandleAvgPeriod(Near)) + 5
    let lookback = 1_usize.max(NEAR.avg_period) + 5;
    if len <= lookback { return Ok(output); }

    // Initialize Near sum for bar (start - 3), i.e. the "2nd candle" at start
    let mut near_sum = 0.0;
    let near_bar = lookback - 3; // the 2nd bar of the pattern at first evaluation
    if NEAR.avg_period > 0 && near_bar >= NEAR.avg_period {
        for j in (near_bar - NEAR.avg_period)..near_bar {
            near_sum += cr(NEAR, open, high, low, close, j);
        }
    }

    let mut pattern_idx: i32 = -10; // no active pattern
    let mut pattern_result: i32 = 0;

    for i in lookback..len {
        // C TA-Lib indices: i is current bar
        // Pattern: bar[i-3] contains bar[i-2], bar[i-2] contains bar[i-1]
        // Then bar[i] breaks out
        if high[i-1] < high[i-2] && low[i-1] > low[i-2]   // bar[i-1] inside bar[i-2]
            && high[i-2] < high[i-3] && low[i-2] > low[i-3] // bar[i-2] inside bar[i-3]
        {
            let near_avg = ca(NEAR, near_sum, open, high, low, close, i - 2);
            // Bullish: bar[i] breaks down (lower high AND lower low)
            if high[i] < high[i-1] && low[i] < low[i-1]
                // 2nd bar close near the low
                && close[i-2] <= low[i-2] + near_avg
            {
                pattern_result = 100;
                pattern_idx = i as i32;
            }
            // Bearish: bar[i] breaks up (higher high AND higher low)
            else if high[i] > high[i-1] && low[i] > low[i-1]
                // 2nd bar close near the high
                && close[i-2] >= high[i-2] - near_avg
            {
                pattern_result = -100;
                pattern_idx = i as i32;
            }
        }

        // Confirmation: within 3 bars of pattern
        if pattern_idx >= 0 && (i as i32) <= pattern_idx + 3 {
            if pattern_result > 0 && close[i] > high[(pattern_idx - 1) as usize] {
                output[i] = pattern_result + 100;
                pattern_idx = -10;
            } else if pattern_result < 0 && close[i] < low[(pattern_idx - 1) as usize] {
                output[i] = pattern_result - 100;
                pattern_idx = -10;
            }
        }

        // Update Near sum (for the "2nd bar" position, which is i-2)
        if NEAR.avg_period > 0 && (i - 2) >= NEAR.avg_period {
            near_sum += cr(NEAR, open, high, low, close, i - 2)
                - cr(NEAR, open, high, low, close, i - 2 - NEAR.avg_period);
        }
    }
    Ok(output)
}

/// CDL_HOMINGPIGEON
pub fn cdl_homingpigeon(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == -1
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) <= ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && open[i] < open[i-1]
            && close[i] > close[i-1]
        {
            output[i] = 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_INNECK
pub fn cdl_inneck(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut equal_sum = 0.0;
    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        // 1st: long black
        if candle_color(open[i-1], close[i-1]) == -1
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum, open, high, low, close, i-1)
            // 2nd: white, opens below prev low
            && candle_color(open[i], close[i]) == 1
            && open[i] < low[i-1]
            // close slightly into prev body: >= prev close and <= prev close + Equal avg
            && close[i] >= close[i-1]
            && close[i] <= close[i-1] + ca(EQUAL, equal_sum, open, high, low, close, i-1)
        {
            output[i] = -100;
        }
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
        body_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_INVERTEDHAMMER
pub fn cdl_invertedhammer(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_SHORT.avg_period, SHADOW_LONG.avg_period, SHADOW_VERY_SHORT.avg_period].iter().max().unwrap() + 1;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_long_sum = 0.0;
    let mut shadow_vs_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_LONG, shadow_long_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum, open, high, low, close, i)
            && real_body_gap_down(open, close, i, i-1)
        {
            output[i] = 100;
        }
        if BODY_SHORT.avg_period > 0 { body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period); }
        if SHADOW_VERY_SHORT.avg_period > 0 { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period); }
    }
    Ok(output)
}

/// CDL_KICKING
pub fn cdl_kicking(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = SHADOW_VERY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = [0.0f64; 2];
    let mut body_sum = [0.0f64; 2];
    let start = lookback;
    for i in (start - 1 - SHADOW_VERY_SHORT.avg_period)..(start - 1) { shadow_sum[1] += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum[0] += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum[1] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_LONG.avg_period)..start { body_sum[0] += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        let color_prev = candle_color(open[i-1], close[i-1]);
        let color_curr = candle_color(open[i], close[i]);
        if color_prev != color_curr
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum[1], open, high, low, close, i-1)
            && upper_shadow(open[i-1], high[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && lower_shadow(open[i-1], low[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && real_body(open[i], close[i]) > ca(BODY_LONG, body_sum[0], open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i)
        {
            // Gap: black then white = bullish, white then black = bearish
            if color_prev == -1 && color_curr == 1 && open[i] > open[i-1] {
                output[i] = 100;
            } else if color_prev == 1 && color_curr == -1 && open[i] < open[i-1] {
                output[i] = -100;
            }
        }
        shadow_sum[1] += cr(SHADOW_VERY_SHORT, open, high, low, close, i-1) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - 1 - SHADOW_VERY_SHORT.avg_period);
        shadow_sum[0] += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
        body_sum[1] += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_sum[0] += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_KICKINGBYLENGTH
pub fn cdl_kickingbylength(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = SHADOW_VERY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = [0.0f64; 2];
    let mut body_sum = [0.0f64; 2];
    let start = lookback;
    for i in (start - 1 - SHADOW_VERY_SHORT.avg_period)..(start - 1) { shadow_sum[1] += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum[0] += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum[1] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_LONG.avg_period)..start { body_sum[0] += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        let color_prev = candle_color(open[i-1], close[i-1]);
        let color_curr = candle_color(open[i], close[i]);
        if color_prev != color_curr
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum[1], open, high, low, close, i-1)
            && upper_shadow(open[i-1], high[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && lower_shadow(open[i-1], low[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && real_body(open[i], close[i]) > ca(BODY_LONG, body_sum[0], open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i)
        {
            // Gap check
            let has_gap = (color_prev == -1 && color_curr == 1 && open[i] > open[i-1])
                || (color_prev == 1 && color_curr == -1 && open[i] < open[i-1]);
            if has_gap {
                // Signal direction based on longer marubozu
                if real_body(open[i], close[i]) >= real_body(open[i-1], close[i-1]) {
                    output[i] = color_curr * 100;
                } else {
                    output[i] = color_prev * 100;
                }
            }
        }
        shadow_sum[1] += cr(SHADOW_VERY_SHORT, open, high, low, close, i-1) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - 1 - SHADOW_VERY_SHORT.avg_period);
        shadow_sum[0] += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
        body_sum[1] += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_sum[0] += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_MATCHINGLOW
pub fn cdl_matchinglow(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = EQUAL.avg_period + 1;
    if len <= lookback { return Ok(output); }

    let mut equal_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == -1
            && (close[i] - close[i-1]).abs() <= ca(EQUAL, equal_sum, open, high, low, close, i-1)
        {
            output[i] = 100;
        }
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
    }
    Ok(output)
}

/// CDL_ONNECK
pub fn cdl_onneck(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut equal_sum = 0.0;
    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == -1
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum, open, high, low, close, i-1)
            && candle_color(open[i], close[i]) == 1
            && open[i] < low[i-1]
            && (close[i] - low[i-1]).abs() <= ca(EQUAL, equal_sum, open, high, low, close, i-1)
        {
            output[i] = -100;
        }
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
        body_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_PIERCING
pub fn cdl_piercing(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period + 1;
    if len <= lookback { return Ok(output); }

    let mut body_sum = [0.0f64; 2];
    let start = lookback;
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum[1] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_LONG.avg_period)..start { body_sum[0] += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == -1
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum[1], open, high, low, close, i-1)
            && candle_color(open[i], close[i]) == 1
            && real_body(open[i], close[i]) > ca(BODY_LONG, body_sum[0], open, high, low, close, i)
            && open[i] < low[i-1]
            && close[i] < open[i-1]
            && close[i] > close[i-1] + real_body(open[i-1], close[i-1]) * 0.5
        {
            output[i] = 100;
        }
        body_sum[1] += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_sum[0] += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_SEPARATINGLINES
pub fn cdl_separatinglines(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[SHADOW_VERY_SHORT.avg_period, BODY_LONG.avg_period, EQUAL.avg_period].iter().max().unwrap() + 1;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = 0.0;
    let mut body_sum = 0.0;
    let mut equal_sum = 0.0;
    let start = lookback;
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - BODY_LONG.avg_period)..start { body_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }

    for i in start..len {
        let color_prev = candle_color(open[i-1], close[i-1]);
        let color_curr = candle_color(open[i], close[i]);
        if color_prev != color_curr
            && (open[i] - open[i-1]).abs() <= ca(EQUAL, equal_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) > ca(BODY_LONG, body_sum, open, high, low, close, i)
        {
            // Bullish: very short lower shadow, bearish: very short upper shadow
            if color_curr == 1 && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i) {
                output[i] = 100;
            } else if color_curr == -1 && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i) {
                output[i] = -100;
            }
        }
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
        body_sum += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
    }
    Ok(output)
}

/// CDL_SHOOTINGSTAR
pub fn cdl_shootingstar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_SHORT.avg_period, SHADOW_LONG.avg_period, SHADOW_VERY_SHORT.avg_period].iter().max().unwrap() + 1;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_long_sum = 0.0;
    let mut shadow_vs_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_SHORT.avg_period)..start { body_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) < ca(BODY_SHORT, body_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_LONG, shadow_long_sum, open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum, open, high, low, close, i)
            && real_body_gap_up(open, close, i, i-1)
        {
            output[i] = -100;
        }
        if BODY_SHORT.avg_period > 0 { body_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period); }
        if SHADOW_VERY_SHORT.avg_period > 0 { shadow_vs_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period); }
    }
    Ok(output)
}

/// CDL_STICKSANDWICH
pub fn cdl_sticksandwich(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = EQUAL.avg_period + 2;
    if len <= lookback { return Ok(output); }

    let mut equal_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - EQUAL.avg_period)..(start - 2) { equal_sum += cr(EQUAL, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == 1
            && candle_color(open[i], close[i]) == -1
            && low[i-1] > close[i-2]
            && (close[i] - close[i-2]).abs() <= ca(EQUAL, equal_sum, open, high, low, close, i-2)
        {
            output[i] = 100;
        }
        equal_sum += cr(EQUAL, open, high, low, close, i-2) - cr(EQUAL, open, high, low, close, i - 2 - EQUAL.avg_period);
    }
    Ok(output)
}

/// CDL_THRUSTING
pub fn cdl_thrusting(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    if len <= lookback { return Ok(output); }

    let mut equal_sum = 0.0;
    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == -1
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_sum, open, high, low, close, i-1)
            && candle_color(open[i], close[i]) == 1
            && open[i] < low[i-1]
            && close[i] > close[i-1] + ca(EQUAL, equal_sum, open, high, low, close, i-1)
            && close[i] <= close[i-1] + real_body(open[i-1], close[i-1]) * 0.5
        {
            output[i] = -100;
        }
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
        body_sum += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_BELTHOLD
pub fn cdl_belthold(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - BODY_LONG.avg_period)..start { body_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i], close[i]) > ca(BODY_LONG, body_sum, open, high, low, close, i) {
            if candle_color(open[i], close[i]) == 1
                && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            {
                output[i] = 100;
            } else if candle_color(open[i], close[i]) == -1
                && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i)
            {
                output[i] = -100;
            }
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

// ========== Three+ candle patterns ==========

/// CDL_3BLACKCROWS
pub fn cdl_3blackcrows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = SHADOW_VERY_SHORT.avg_period + 3;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = [0.0f64; 3];
    let start = lookback;
    for k in 0..3 {
        let bar_offset = start - 3 + k;
        if bar_offset >= SHADOW_VERY_SHORT.avg_period {
            for j in (bar_offset - SHADOW_VERY_SHORT.avg_period)..bar_offset {
                shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, j);
            }
        }
    }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == -1
            && close[i-1] < close[i-2] && close[i] < close[i-1]
            && open[i-2] <= open[i-3].max(close[i-3])
            && open[i-1] <= open[i-2] && open[i-1] >= close[i-2]
            && open[i] <= open[i-1] && open[i] >= close[i-1]
            && lower_shadow(open[i-2], low[i-2], close[i-2]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i-2)
            && lower_shadow(open[i-1], low[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[2], open, high, low, close, i)
        {
            output[i] = -100;
        }
        for k in 0..3 {
            let bar = i - 2 + k;
            if bar >= SHADOW_VERY_SHORT.avg_period {
                shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, bar)
                    - cr(SHADOW_VERY_SHORT, open, high, low, close, bar - SHADOW_VERY_SHORT.avg_period);
            }
        }
    }
    Ok(output)
}

/// CDL_3INSIDE
pub fn cdl_3inside(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_SHORT.avg_period)..(start - 1) { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_SHORT, body_short_sum, open, high, low, close, i-1)
            && open[i-1].max(close[i-1]) < open[i-2].max(close[i-2])
            && open[i-1].min(close[i-1]) > open[i-2].min(close[i-2])
            && ((candle_color(open[i-2], close[i-2]) == 1 && candle_color(open[i], close[i]) == -1 && close[i] < open[i-2])
                || (candle_color(open[i-2], close[i-2]) == -1 && candle_color(open[i], close[i]) == 1 && close[i] > open[i-2]))
        {
            output[i] = -candle_color(open[i-2], close[i-2]) * 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i-1) - cr(BODY_SHORT, open, high, low, close, i - 1 - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_3LINESTRIKE
pub fn cdl_3linestrike(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = NEAR.avg_period + 3;
    if len <= lookback { return Ok(output); }

    let mut near_sum = [0.0f64; 4];
    let start = lookback;
    // Init near sums for bars i-3 and i-2
    for k in [2usize, 3] {
        let bar = start - k;
        if bar >= NEAR.avg_period {
            for j in (bar - NEAR.avg_period)..bar {
                near_sum[k] += cr(NEAR, open, high, low, close, j);
            }
        }
    }

    for i in start..len {
        let c3 = candle_color(open[i-3], close[i-3]);
        let c2 = candle_color(open[i-2], close[i-2]);
        let c1 = candle_color(open[i-1], close[i-1]);
        let c0 = candle_color(open[i], close[i]);

        if c3 == c2 && c2 == c1 && c0 != c1 {
            // Three same-color, 4th opposite
            let progressive = if c3 == 1 {
                close[i-2] > close[i-3] && close[i-1] > close[i-2]
            } else {
                close[i-2] < close[i-3] && close[i-1] < close[i-2]
            };
            let opens_near = if c3 == 1 {
                open[i-2] >= open[i-3].min(close[i-3])
                    && open[i-2] <= close[i-3] + ca(NEAR, near_sum[3], open, high, low, close, i-3)
                    && open[i-1] >= open[i-2].min(close[i-2])
                    && open[i-1] <= close[i-2] + ca(NEAR, near_sum[2], open, high, low, close, i-2)
            } else {
                open[i-2] <= open[i-3].max(close[i-3])
                    && open[i-2] >= close[i-3] - ca(NEAR, near_sum[3], open, high, low, close, i-3)
                    && open[i-1] <= open[i-2].max(close[i-2])
                    && open[i-1] >= close[i-2] - ca(NEAR, near_sum[2], open, high, low, close, i-2)
            };
            let strike = if c3 == 1 {
                open[i] >= close[i-1] && close[i] <= open[i-3]
            } else {
                open[i] <= close[i-1] && close[i] >= open[i-3]
            };
            if progressive && opens_near && strike {
                output[i] = c3 * 100;
            }
        }
        // Update near sums
        for k in [2usize, 3] {
            let bar = i - k;
            if bar >= NEAR.avg_period && NEAR.avg_period > 0 {
                near_sum[k] += cr(NEAR, open, high, low, close, bar) - cr(NEAR, open, high, low, close, bar - NEAR.avg_period);
            }
        }
    }
    Ok(output)
}

/// CDL_3OUTSIDE
pub fn cdl_3outside(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    // lookback = 3
    if len < 3 { return Ok(output); }

    for i in 2..len {
        // Bullish: 1st black, 2nd white engulfs, 3rd closes higher
        if candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == 1
            && close[i-1] >= open[i-2]
            && open[i-1] <= close[i-2]
            && close[i] > close[i-1]
        {
            output[i] = 100;
        }
        // Bearish: 1st white, 2nd black engulfs, 3rd closes lower
        else if candle_color(open[i-2], close[i-2]) == 1
            && candle_color(open[i-1], close[i-1]) == -1
            && open[i-1] >= close[i-2]
            && close[i-1] <= open[i-2]
            && close[i] < close[i-1]
        {
            output[i] = -100;
        }
    }
    Ok(output)
}

/// CDL_3STARSINSOUTH
pub fn cdl_3starsinsouth(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[SHADOW_VERY_SHORT.avg_period, SHADOW_LONG.avg_period, BODY_LONG.avg_period, BODY_SHORT.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut shadow_long_sum = 0.0;
    let mut shadow_vs_sum = [0.0f64; 2]; // for 2nd and 3rd candles
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    // SHADOW_LONG avg_period = 0, no init
    for i in (start - 1 - SHADOW_VERY_SHORT.avg_period)..(start - 1) { shadow_vs_sum[0] += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - SHADOW_VERY_SHORT.avg_period)..start { shadow_vs_sum[1] += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == -1
            // 1st: long body, long lower shadow
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && lower_shadow(open[i-2], low[i-2], close[i-2]) > ca(SHADOW_LONG, shadow_long_sum, open, high, low, close, i-2)
            // 2nd: body inside 1st, low < 1st low, short lower shadow
            && open[i-1].min(close[i-1]) > open[i-2].min(close[i-2])
            && open[i-1].max(close[i-1]) < open[i-2].max(close[i-2])
            && low[i-1] < low[i-2]
            // 3rd: short body, short shadows, within 2nd range
            && real_body(open[i], close[i]) < ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum[1], open, high, low, close, i)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_vs_sum[1], open, high, low, close, i)
            && low[i] > low[i-1] && high[i] < high[i-1]
        {
            output[i] = 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        shadow_vs_sum[0] += cr(SHADOW_VERY_SHORT, open, high, low, close, i-1) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - 1 - SHADOW_VERY_SHORT.avg_period);
        shadow_vs_sum[1] += cr(SHADOW_VERY_SHORT, open, high, low, close, i) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - SHADOW_VERY_SHORT.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_3WHITESOLDIERS
pub fn cdl_3whitesoldiers(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[SHADOW_VERY_SHORT.avg_period, BODY_SHORT.avg_period, FAR.avg_period, NEAR.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = [0.0f64; 3];
    let mut near_sum = [0.0f64; 3];
    let mut far_sum = [0.0f64; 3];
    let mut body_short_sum = 0.0;
    let start = lookback;

    for k in 0..3 {
        let bar = start - 2 + k;
        if bar >= SHADOW_VERY_SHORT.avg_period {
            for j in (bar - SHADOW_VERY_SHORT.avg_period)..bar { shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, j); }
        }
        if bar >= NEAR.avg_period {
            for j in (bar - NEAR.avg_period)..bar { near_sum[k] += cr(NEAR, open, high, low, close, j); }
        }
        if bar >= FAR.avg_period {
            for j in (bar - FAR.avg_period)..bar { far_sum[k] += cr(FAR, open, high, low, close, j); }
        }
    }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == 1
            && candle_color(open[i-1], close[i-1]) == 1
            && candle_color(open[i], close[i]) == 1
            && close[i-1] > close[i-2] && close[i] > close[i-1]
            // Short upper shadows
            && upper_shadow(open[i-2], high[i-2], close[i-2]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i-2)
            && upper_shadow(open[i-1], high[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && upper_shadow(open[i], high[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[2], open, high, low, close, i)
            // Opens within or near previous body
            && open[i-1] > open[i-2] && open[i-1] <= close[i-2] + ca(NEAR, near_sum[1], open, high, low, close, i-1)
            && open[i] > open[i-1] && open[i] <= close[i-1] + ca(NEAR, near_sum[2], open, high, low, close, i)
            // Bodies not far shorter than prior
            && real_body(open[i-1], close[i-1]) > real_body(open[i-2], close[i-2]) - ca(FAR, far_sum[1], open, high, low, close, i-1)
            && real_body(open[i], close[i]) > real_body(open[i-1], close[i-1]) - ca(FAR, far_sum[2], open, high, low, close, i)
            // Last body not short
            && real_body(open[i], close[i]) > ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
        {
            output[i] = 100;
        }
        for k in 0..3 {
            let bar = i - 2 + k;
            if SHADOW_VERY_SHORT.avg_period > 0 && bar >= SHADOW_VERY_SHORT.avg_period {
                shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, bar) - cr(SHADOW_VERY_SHORT, open, high, low, close, bar - SHADOW_VERY_SHORT.avg_period);
            }
            if NEAR.avg_period > 0 && bar >= NEAR.avg_period {
                near_sum[k] += cr(NEAR, open, high, low, close, bar) - cr(NEAR, open, high, low, close, bar - NEAR.avg_period);
            }
            if FAR.avg_period > 0 && bar >= FAR.avg_period {
                far_sum[k] += cr(FAR, open, high, low, close, bar) - cr(FAR, open, high, low, close, bar - FAR.avg_period);
            }
        }
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_ABANDONEDBABY
pub fn cdl_abandonedbaby(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.3;
    let lookback = *[BODY_DOJI.avg_period, BODY_LONG.avg_period, BODY_SHORT.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_doji_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_DOJI.avg_period)..(start - 1) { body_doji_sum += cr(BODY_DOJI, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_DOJI, body_doji_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) > ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
        {
            // Bullish: 1st black, gap down doji, gap up white
            if candle_color(open[i-2], close[i-2]) == -1
                && candle_color(open[i], close[i]) == 1
                && high[i-1] < low[i-2]
                && low[i] > high[i-1]
                && close[i] > close[i-2] + real_body(open[i-2], close[i-2]) * penetration
            {
                output[i] = 100;
            }
            // Bearish: 1st white, gap up doji, gap down black
            else if candle_color(open[i-2], close[i-2]) == 1
                && candle_color(open[i], close[i]) == -1
                && low[i-1] > high[i-2]
                && high[i] < low[i-1]
                && close[i] < close[i-2] - real_body(open[i-2], close[i-2]) * penetration
            {
                output[i] = -100;
            }
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_doji_sum += cr(BODY_DOJI, open, high, low, close, i-1) - cr(BODY_DOJI, open, high, low, close, i - 1 - BODY_DOJI.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_ADVANCEBLOCK
pub fn cdl_advanceblock(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[SHADOW_LONG.avg_period, SHADOW_SHORT.avg_period, FAR.avg_period, NEAR.avg_period, BODY_LONG.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut shadow_short_sum = [0.0f64; 3];
    let mut shadow_long_sum = [0.0f64; 3];
    let mut near_sum = [0.0f64; 3];
    let mut far_sum = [0.0f64; 3];
    let mut body_long_sum = 0.0;
    let start = lookback;

    for k in 0..3 {
        let bar = start - 2 + k;
        // SHADOW_LONG and SHADOW_SHORT init
        if SHADOW_SHORT.avg_period > 0 && bar >= SHADOW_SHORT.avg_period {
            for j in (bar - SHADOW_SHORT.avg_period)..bar { shadow_short_sum[k] += cr(SHADOW_SHORT, open, high, low, close, j); }
        }
        // SHADOW_LONG avg_period=0
        if NEAR.avg_period > 0 && bar >= NEAR.avg_period {
            for j in (bar - NEAR.avg_period)..bar { near_sum[k] += cr(NEAR, open, high, low, close, j); }
        }
        if FAR.avg_period > 0 && bar >= FAR.avg_period {
            for j in (bar - FAR.avg_period)..bar { far_sum[k] += cr(FAR, open, high, low, close, j); }
        }
    }
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == 1
            && candle_color(open[i-1], close[i-1]) == 1
            && candle_color(open[i], close[i]) == 1
            && close[i-1] > close[i-2] && close[i] > close[i-1]
            // Opens within/near previous body
            && open[i-1] > open[i-2] && open[i-1] <= close[i-2] + ca(NEAR, near_sum[1], open, high, low, close, i-1)
            && open[i] > open[i-1] && open[i] <= close[i-1] + ca(NEAR, near_sum[2], open, high, low, close, i)
            // 1st: long body, short upper shadow
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && upper_shadow(open[i-2], high[i-2], close[i-2]) < ca(SHADOW_SHORT, shadow_short_sum[0], open, high, low, close, i-2)
        {
            // Weakness: bodies getting smaller and/or shadows getting longer
            let weakness =
                (real_body(open[i-1], close[i-1]) < real_body(open[i-2], close[i-2]) - ca(FAR, far_sum[1], open, high, low, close, i-1)
                    && real_body(open[i], close[i]) < real_body(open[i-1], close[i-1]) + ca(NEAR, near_sum[2], open, high, low, close, i))
                || (real_body(open[i], close[i]) < real_body(open[i-1], close[i-1])
                    && real_body(open[i-1], close[i-1]) < real_body(open[i-2], close[i-2])
                    && (upper_shadow(open[i], high[i], close[i]) > ca(SHADOW_LONG, shadow_long_sum[2], open, high, low, close, i)
                        || upper_shadow(open[i-1], high[i-1], close[i-1]) > ca(SHADOW_LONG, shadow_long_sum[1], open, high, low, close, i-1)))
                || (real_body(open[i], close[i]) < real_body(open[i-1], close[i-1]) - ca(FAR, far_sum[2], open, high, low, close, i));

            if weakness {
                output[i] = -100;
            }
        }
        // Update sums
        for k in 0..3 {
            let bar = i - 2 + k;
            if SHADOW_SHORT.avg_period > 0 && bar >= SHADOW_SHORT.avg_period {
                shadow_short_sum[k] += cr(SHADOW_SHORT, open, high, low, close, bar) - cr(SHADOW_SHORT, open, high, low, close, bar - SHADOW_SHORT.avg_period);
            }
            if NEAR.avg_period > 0 && bar >= NEAR.avg_period {
                near_sum[k] += cr(NEAR, open, high, low, close, bar) - cr(NEAR, open, high, low, close, bar - NEAR.avg_period);
            }
            if FAR.avg_period > 0 && bar >= FAR.avg_period {
                far_sum[k] += cr(FAR, open, high, low, close, bar) - cr(FAR, open, high, low, close, bar - FAR.avg_period);
            }
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_BREAKAWAY
pub fn cdl_breakaway(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_LONG.avg_period + 4;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 4 - BODY_LONG.avg_period)..(start - 4) { body_sum += cr(BODY_LONG, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-4], close[i-4]) > ca(BODY_LONG, body_sum, open, high, low, close, i-4)
            && candle_color(open[i-4], close[i-4]) == candle_color(open[i-3], close[i-3])
            && candle_color(open[i-3], close[i-3]) == candle_color(open[i-1], close[i-1])
            && candle_color(open[i-1], close[i-1]) == -candle_color(open[i], close[i])
        {
            // Bearish first (black): gap down, progressive lower H/L, 5th closes in gap
            if candle_color(open[i-4], close[i-4]) == -1
                && real_body_gap_down(open, close, i-3, i-4)
                && high[i-2] < high[i-3] && low[i-2] < low[i-3]
                && high[i-1] < high[i-2] && low[i-1] < low[i-2]
                && close[i] > open[i-3] && close[i] < close[i-4]
            {
                output[i] = candle_color(open[i], close[i]) * 100;
            }
            // Bullish first (white): gap up, progressive higher H/L, 5th closes in gap
            else if candle_color(open[i-4], close[i-4]) == 1
                && real_body_gap_up(open, close, i-3, i-4)
                && high[i-2] > high[i-3] && low[i-2] > low[i-3]
                && high[i-1] > high[i-2] && low[i-1] > low[i-2]
                && close[i] < open[i-3] && close[i] > close[i-4]
            {
                output[i] = candle_color(open[i], close[i]) * 100;
            }
        }
        body_sum += cr(BODY_LONG, open, high, low, close, i-4) - cr(BODY_LONG, open, high, low, close, i - 4 - BODY_LONG.avg_period);
    }
    Ok(output)
}

/// CDL_CONCEALBABYSWALL
pub fn cdl_concealbabyswall(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = SHADOW_VERY_SHORT.avg_period + 3;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = [0.0f64; 4];
    let start = lookback;
    for k in 0..4 {
        let bar = start - 3 + k;
        if bar >= SHADOW_VERY_SHORT.avg_period {
            for j in (bar - SHADOW_VERY_SHORT.avg_period)..bar {
                shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, j);
            }
        }
    }

    for i in start..len {
        if candle_color(open[i-3], close[i-3]) == -1
            && candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == -1
            // 1st and 2nd: marubozu (very short shadows)
            && upper_shadow(open[i-3], high[i-3], close[i-3]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i-3)
            && lower_shadow(open[i-3], low[i-3], close[i-3]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i-3)
            && upper_shadow(open[i-2], high[i-2], close[i-2]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-2)
            && lower_shadow(open[i-2], low[i-2], close[i-2]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-2)
            // 3rd: gaps down, upper shadow into 2nd body
            && real_body_gap_down(open, close, i-1, i-2)
            && high[i-1] > close[i-2]
            // 4th: engulfs 3rd including shadows
            && open[i] >= high[i-1] && close[i] <= low[i-1]
        {
            output[i] = 100;
        }
        for k in 0..4 {
            let bar = i - 3 + k;
            if SHADOW_VERY_SHORT.avg_period > 0 && bar >= SHADOW_VERY_SHORT.avg_period {
                shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, bar)
                    - cr(SHADOW_VERY_SHORT, open, high, low, close, bar - SHADOW_VERY_SHORT.avg_period);
            }
        }
    }
    Ok(output)
}

/// CDL_EVENINGDOJISTAR
pub fn cdl_eveningdojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.3;
    let lookback = *[BODY_DOJI.avg_period, BODY_LONG.avg_period, BODY_SHORT.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_doji_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_DOJI.avg_period)..(start - 1) { body_doji_sum += cr(BODY_DOJI, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == 1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_DOJI, body_doji_sum, open, high, low, close, i-1)
            && real_body_gap_up(open, close, i-1, i-2)
            && candle_color(open[i], close[i]) == -1
            && real_body(open[i], close[i]) > ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && close[i] < close[i-2] - real_body(open[i-2], close[i-2]) * penetration
        {
            output[i] = -100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_doji_sum += cr(BODY_DOJI, open, high, low, close, i-1) - cr(BODY_DOJI, open, high, low, close, i - 1 - BODY_DOJI.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_EVENINGSTAR
pub fn cdl_eveningstar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.3;
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let mut body_short_sum2 = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_SHORT.avg_period)..(start - 1) { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum2 += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == 1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_SHORT, body_short_sum, open, high, low, close, i-1)
            && real_body_gap_up(open, close, i-1, i-2)
            && candle_color(open[i], close[i]) == -1
            && real_body(open[i], close[i]) > ca(BODY_SHORT, body_short_sum2, open, high, low, close, i)
            && close[i] < close[i-2] - real_body(open[i-2], close[i-2]) * penetration
        {
            output[i] = -100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i-1) - cr(BODY_SHORT, open, high, low, close, i - 1 - BODY_SHORT.avg_period);
        body_short_sum2 += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_GAPSIDESIDEWHITE
pub fn cdl_gapsidesidewhite(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = NEAR.avg_period.max(EQUAL.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut near_sum = 0.0;
    let mut equal_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - NEAR.avg_period)..(start - 1) { near_sum += cr(NEAR, open, high, low, close, i); }
    for i in (start - 1 - EQUAL.avg_period)..(start - 1) { equal_sum += cr(EQUAL, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-1], close[i-1]) == 1
            && candle_color(open[i], close[i]) == 1
            && (real_body(open[i-1], close[i-1]) - real_body(open[i], close[i])).abs() < ca(NEAR, near_sum, open, high, low, close, i-1)
            && (open[i-1] - open[i]).abs() < ca(EQUAL, equal_sum, open, high, low, close, i-1)
        {
            // Upside gap
            if real_body_gap_up(open, close, i-1, i-2) {
                output[i] = 100;
            }
            // Downside gap
            else if real_body_gap_down(open, close, i-1, i-2) {
                output[i] = -100;
            }
        }
        near_sum += cr(NEAR, open, high, low, close, i-1) - cr(NEAR, open, high, low, close, i - 1 - NEAR.avg_period);
        equal_sum += cr(EQUAL, open, high, low, close, i-1) - cr(EQUAL, open, high, low, close, i - 1 - EQUAL.avg_period);
    }
    Ok(output)
}

/// CDL_IDENTICAL3CROWS
pub fn cdl_identical3crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = SHADOW_VERY_SHORT.avg_period.max(EQUAL.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = [0.0f64; 3];
    let mut equal_sum = [0.0f64; 3];
    let start = lookback;
    for k in 0..3 {
        let bar = start - 2 + k;
        if bar >= SHADOW_VERY_SHORT.avg_period {
            for j in (bar - SHADOW_VERY_SHORT.avg_period)..bar { shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, j); }
        }
        if k < 2 && bar >= EQUAL.avg_period {
            for j in (bar - EQUAL.avg_period)..bar { equal_sum[k] += cr(EQUAL, open, high, low, close, j); }
        }
    }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == -1
            && candle_color(open[i], close[i]) == -1
            && close[i-1] < close[i-2] && close[i] < close[i-1]
            // Very short lower shadows
            && lower_shadow(open[i-2], low[i-2], close[i-2]) < ca(SHADOW_VERY_SHORT, shadow_sum[0], open, high, low, close, i-2)
            && lower_shadow(open[i-1], low[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum[1], open, high, low, close, i-1)
            && lower_shadow(open[i], low[i], close[i]) < ca(SHADOW_VERY_SHORT, shadow_sum[2], open, high, low, close, i)
            // Each opens equal to prior close
            && (open[i-1] - close[i-2]).abs() <= ca(EQUAL, equal_sum[0], open, high, low, close, i-2)
            && (open[i] - close[i-1]).abs() <= ca(EQUAL, equal_sum[1], open, high, low, close, i-1)
        {
            output[i] = -100;
        }
        for k in 0..3 {
            let bar = i - 2 + k;
            if SHADOW_VERY_SHORT.avg_period > 0 && bar >= SHADOW_VERY_SHORT.avg_period {
                shadow_sum[k] += cr(SHADOW_VERY_SHORT, open, high, low, close, bar) - cr(SHADOW_VERY_SHORT, open, high, low, close, bar - SHADOW_VERY_SHORT.avg_period);
            }
        }
        for k in 0..2 {
            let bar = i - 2 + k;
            if EQUAL.avg_period > 0 && bar >= EQUAL.avg_period {
                equal_sum[k] += cr(EQUAL, open, high, low, close, bar) - cr(EQUAL, open, high, low, close, bar - EQUAL.avg_period);
            }
        }
    }
    Ok(output)
}

/// CDL_LADDERBOTTOM
pub fn cdl_ladderbottom(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = SHADOW_VERY_SHORT.avg_period + 4;
    if len <= lookback { return Ok(output); }

    let mut shadow_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - SHADOW_VERY_SHORT.avg_period)..(start - 1) { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-4], close[i-4]) == -1
            && candle_color(open[i-3], close[i-3]) == -1
            && candle_color(open[i-2], close[i-2]) == -1
            && candle_color(open[i-1], close[i-1]) == -1
            && close[i-3] < close[i-4] && close[i-2] < close[i-3]
            // 4th: upper shadow exceeds very short
            && upper_shadow(open[i-1], high[i-1], close[i-1]) > ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i-1)
            // 5th: white, opens above 4th open, closes above 4th high
            && candle_color(open[i], close[i]) == 1
            && open[i] > open[i-1]
            && close[i] > high[i-1]
        {
            output[i] = 100;
        }
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i-1) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - 1 - SHADOW_VERY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_MATHOLD
pub fn cdl_mathold(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.5;
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 4;
    if len <= lookback { return Ok(output); }

    let mut body_sum = [0.0f64; 5];
    let start = lookback;
    // Init long body sum for i-4, short for i-3..i-1, long for i
    for i in (start - 4 - BODY_LONG.avg_period)..(start - 4) { body_sum[4] += cr(BODY_LONG, open, high, low, close, i); }
    for k in 1..4 {
        let bar = start - 4 + k;
        for j in (bar - BODY_SHORT.avg_period)..bar { body_sum[4 - k] += cr(BODY_SHORT, open, high, low, close, j); }
    }

    for i in start..len {
        if real_body(open[i-4], close[i-4]) > ca(BODY_LONG, body_sum[4], open, high, low, close, i-4)
            && real_body(open[i-3], close[i-3]) < ca(BODY_SHORT, body_sum[3], open, high, low, close, i-3)
            && real_body(open[i-2], close[i-2]) < ca(BODY_SHORT, body_sum[2], open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) < ca(BODY_SHORT, body_sum[1], open, high, low, close, i-1)
            // white, black, ?, ?, white
            && candle_color(open[i-4], close[i-4]) == 1
            && candle_color(open[i-3], close[i-3]) == -1
            && candle_color(open[i], close[i]) == 1
            // upside gap 1st to 2nd
            && real_body_gap_up(open, close, i-3, i-4)
            // 3rd and 4th hold within 1st close
            && open[i-2].min(close[i-2]) < close[i-4]
            && open[i-1].min(close[i-1]) < close[i-4]
            // penetration check
            && open[i-2].min(close[i-2]) > close[i-4] - real_body(open[i-4], close[i-4]) * penetration
            && open[i-1].min(close[i-1]) > close[i-4] - real_body(open[i-4], close[i-4]) * penetration
            // 2nd to 4th are falling
            && open[i-2].max(close[i-2]) < open[i-3]
            && open[i-1].max(close[i-1]) < open[i-2].max(close[i-2])
            // 5th opens above prior close, closes above highest reaction high
            && open[i] > close[i-1]
            && close[i] > high[i-3].max(high[i-2]).max(high[i-1])
        {
            output[i] = 100;
        }
        body_sum[4] += cr(BODY_LONG, open, high, low, close, i-4) - cr(BODY_LONG, open, high, low, close, i - 4 - BODY_LONG.avg_period);
        for k in 1..4 {
            let bar = i - 4 + k;
            body_sum[4 - k] += cr(BODY_SHORT, open, high, low, close, bar) - cr(BODY_SHORT, open, high, low, close, bar - BODY_SHORT.avg_period);
        }
    }
    Ok(output)
}

/// CDL_MORNINGDOJISTAR
pub fn cdl_morningdojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.3;
    let lookback = *[BODY_DOJI.avg_period, BODY_LONG.avg_period, BODY_SHORT.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_doji_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_DOJI.avg_period)..(start - 1) { body_doji_sum += cr(BODY_DOJI, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_DOJI, body_doji_sum, open, high, low, close, i-1)
            && real_body_gap_down(open, close, i-1, i-2)
            && candle_color(open[i], close[i]) == 1
            && real_body(open[i], close[i]) > ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && close[i] > close[i-2] + real_body(open[i-2], close[i-2]) * penetration
        {
            output[i] = 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_doji_sum += cr(BODY_DOJI, open, high, low, close, i-1) - cr(BODY_DOJI, open, high, low, close, i - 1 - BODY_DOJI.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_MORNINGSTAR
pub fn cdl_morningstar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let penetration = 0.3;
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let mut body_short_sum2 = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_SHORT.avg_period)..(start - 1) { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum2 += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_SHORT, body_short_sum, open, high, low, close, i-1)
            && real_body_gap_down(open, close, i-1, i-2)
            && candle_color(open[i], close[i]) == 1
            && real_body(open[i], close[i]) > ca(BODY_SHORT, body_short_sum2, open, high, low, close, i)
            && close[i] > close[i-2] + real_body(open[i-2], close[i-2]) * penetration
        {
            output[i] = 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i-1) - cr(BODY_SHORT, open, high, low, close, i - 1 - BODY_SHORT.avg_period);
        body_short_sum2 += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_RISEFALL3METHODS
pub fn cdl_risefall3methods(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 4;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = [0.0f64; 2]; // [0]=i, [1]=i-4
    let mut body_short_sum = [0.0f64; 3]; // for i-3, i-2, i-1
    let start = lookback;
    for i in (start - 4 - BODY_LONG.avg_period)..(start - 4) { body_long_sum[1] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_LONG.avg_period)..start { body_long_sum[0] += cr(BODY_LONG, open, high, low, close, i); }
    for k in 0..3 {
        let bar = start - 3 + k;
        for j in (bar - BODY_SHORT.avg_period)..bar { body_short_sum[k] += cr(BODY_SHORT, open, high, low, close, j); }
    }

    for i in start..len {
        let c4 = candle_color(open[i-4], close[i-4]);
        let c0 = candle_color(open[i], close[i]);
        if real_body(open[i-4], close[i-4]) > ca(BODY_LONG, body_long_sum[1], open, high, low, close, i-4)
            && real_body(open[i], close[i]) > ca(BODY_LONG, body_long_sum[0], open, high, low, close, i)
        {
            // 1st long, 3 short middle, 5th long
            let mid_short = real_body(open[i-3], close[i-3]) < ca(BODY_SHORT, body_short_sum[0], open, high, low, close, i-3)
                && real_body(open[i-2], close[i-2]) < ca(BODY_SHORT, body_short_sum[1], open, high, low, close, i-2)
                && real_body(open[i-1], close[i-1]) < ca(BODY_SHORT, body_short_sum[2], open, high, low, close, i-1);

            if c4 == 1 && mid_short
                && candle_color(open[i-3], close[i-3]) == -1
                && candle_color(open[i-2], close[i-2]) == -1
                && candle_color(open[i-1], close[i-1]) == -1
                && close[i-3] < close[i-4]
                && close[i-2] < close[i-3]
                && close[i-1] < close[i-2]
                && low[i-3] > low[i-4] && low[i-2] > low[i-4] && low[i-1] > low[i-4]
                && high[i-3] < high[i-4] && high[i-2] < high[i-4] && high[i-1] < high[i-4]
                && c0 == 1
                && open[i] > close[i-1]
                && close[i] > close[i-4]
            {
                output[i] = 100;
            } else if c4 == -1 && mid_short
                && candle_color(open[i-3], close[i-3]) == 1
                && candle_color(open[i-2], close[i-2]) == 1
                && candle_color(open[i-1], close[i-1]) == 1
                && close[i-3] > close[i-4]
                && close[i-2] > close[i-3]
                && close[i-1] > close[i-2]
                && high[i-3] < high[i-4] && high[i-2] < high[i-4] && high[i-1] < high[i-4]
                && low[i-3] > low[i-4] && low[i-2] > low[i-4] && low[i-1] > low[i-4]
                && c0 == -1
                && open[i] < close[i-1]
                && close[i] < close[i-4]
            {
                output[i] = -100;
            }
        }
        body_long_sum[1] += cr(BODY_LONG, open, high, low, close, i-4) - cr(BODY_LONG, open, high, low, close, i - 4 - BODY_LONG.avg_period);
        body_long_sum[0] += cr(BODY_LONG, open, high, low, close, i) - cr(BODY_LONG, open, high, low, close, i - BODY_LONG.avg_period);
        for k in 0..3 {
            let bar = i - 3 + k;
            body_short_sum[k] += cr(BODY_SHORT, open, high, low, close, bar) - cr(BODY_SHORT, open, high, low, close, bar - BODY_SHORT.avg_period);
        }
    }
    Ok(output)
}

/// CDL_STALLEDPATTERN
pub fn cdl_stalledpattern(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = *[BODY_LONG.avg_period, BODY_SHORT.avg_period, SHADOW_VERY_SHORT.avg_period, NEAR.avg_period].iter().max().unwrap() + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = [0.0f64; 2]; // for i-2 and i-1
    let mut body_short_sum = 0.0;
    let mut shadow_sum = 0.0;
    let mut near_sum = [0.0f64; 3];
    let start = lookback;

    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum[0] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_LONG.avg_period)..(start - 1) { body_long_sum[1] += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }
    for i in (start - 1 - SHADOW_VERY_SHORT.avg_period)..(start - 1) { shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i); }
    for k in 0..3 {
        let bar = start - 2 + k;
        if NEAR.avg_period > 0 && bar >= NEAR.avg_period {
            for j in (bar - NEAR.avg_period)..bar { near_sum[k] += cr(NEAR, open, high, low, close, j); }
        }
    }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == 1
            && candle_color(open[i-1], close[i-1]) == 1
            && candle_color(open[i], close[i]) == 1
            && close[i-1] > close[i-2] && close[i] > close[i-1]
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum[0], open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) > ca(BODY_LONG, body_long_sum[1], open, high, low, close, i-1)
            && upper_shadow(open[i-1], high[i-1], close[i-1]) < ca(SHADOW_VERY_SHORT, shadow_sum, open, high, low, close, i-1)
            && open[i-1] > open[i-2]
            && open[i-1] <= close[i-2] + ca(NEAR, near_sum[0], open, high, low, close, i-2)
            && real_body(open[i], close[i]) < ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && open[i] >= close[i-1] - real_body(open[i], close[i]) - ca(NEAR, near_sum[1], open, high, low, close, i-1)
        {
            output[i] = -100;
        }
        body_long_sum[0] += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_long_sum[1] += cr(BODY_LONG, open, high, low, close, i-1) - cr(BODY_LONG, open, high, low, close, i - 1 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
        shadow_sum += cr(SHADOW_VERY_SHORT, open, high, low, close, i-1) - cr(SHADOW_VERY_SHORT, open, high, low, close, i - 1 - SHADOW_VERY_SHORT.avg_period);
        for k in 0..3 {
            let bar = i - 2 + k;
            if NEAR.avg_period > 0 && bar >= NEAR.avg_period {
                near_sum[k] += cr(NEAR, open, high, low, close, bar) - cr(NEAR, open, high, low, close, bar - NEAR.avg_period);
            }
        }
    }
    Ok(output)
}

/// CDL_TASUKIGAP
pub fn cdl_tasukigap(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = NEAR.avg_period + 2;
    if len <= lookback { return Ok(output); }

    let mut near_sum = 0.0;
    let start = lookback;
    for i in (start - 1 - NEAR.avg_period)..(start - 1) { near_sum += cr(NEAR, open, high, low, close, i); }

    for i in start..len {
        let c1 = candle_color(open[i-1], close[i-1]);
        let c0 = candle_color(open[i], close[i]);

        // Bodies near same size
        let near_same = (real_body(open[i-1], close[i-1]) - real_body(open[i], close[i])).abs()
            < ca(NEAR, near_sum, open, high, low, close, i-1);

        // Bullish: upside gap, white bar then black bar
        if real_body_gap_up(open, close, i-1, i-2)
            && c1 == 1 && c0 == -1
            && open[i] < close[i-1] && open[i] > open[i-1]
            && close[i] < open[i-1]
            && close[i] > open[i-2].max(close[i-2])
            && near_same
        {
            output[i] = c1 * 100;
        }
        // Bearish: downside gap, black bar then white bar
        else if real_body_gap_down(open, close, i-1, i-2)
            && c1 == -1 && c0 == 1
            && open[i] < open[i-1] && open[i] > close[i-1]
            && close[i] > open[i-1]
            && close[i] < open[i-2].min(close[i-2])
            && near_same
        {
            output[i] = c1 * 100;
        }
        near_sum += cr(NEAR, open, high, low, close, i-1) - cr(NEAR, open, high, low, close, i - 1 - NEAR.avg_period);
    }
    Ok(output)
}

/// CDL_TRISTAR
pub fn cdl_tristar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_DOJI.avg_period + 2;
    if len <= lookback { return Ok(output); }

    let mut body_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_DOJI.avg_period)..(start - 2) { body_sum += cr(BODY_DOJI, open, high, low, close, i); }

    for i in start..len {
        if real_body(open[i-2], close[i-2]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i-2)
            && real_body(open[i-1], close[i-1]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i-1)
            && real_body(open[i], close[i]) <= ca(BODY_DOJI, body_sum, open, high, low, close, i)
        {
            // Bearish: 2nd gaps up
            if real_body_gap_up(open, close, i-1, i-2)
                && !real_body_gap_up(open, close, i, i-1)
            {
                output[i] = -100;
            }
            // Bullish: 2nd gaps down
            else if real_body_gap_down(open, close, i-1, i-2)
                && !real_body_gap_down(open, close, i, i-1)
            {
                output[i] = 100;
            }
        }
        body_sum += cr(BODY_DOJI, open, high, low, close, i-2) - cr(BODY_DOJI, open, high, low, close, i - 2 - BODY_DOJI.avg_period);
    }
    Ok(output)
}

/// CDL_UNIQUE3RIVER
pub fn cdl_unique3river(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - BODY_SHORT.avg_period)..start { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == -1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            // 2nd: black, harami, lower low
            && candle_color(open[i-1], close[i-1]) == -1
            && open[i-1].max(close[i-1]) < open[i-2].max(close[i-2])
            && open[i-1].min(close[i-1]) > open[i-2].min(close[i-2])
            && low[i-1] < low[i-2]
            // 3rd: small white, close <= 2nd close
            && candle_color(open[i], close[i]) == 1
            && real_body(open[i], close[i]) < ca(BODY_SHORT, body_short_sum, open, high, low, close, i)
            && close[i] < close[i-1]
        {
            output[i] = 100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i) - cr(BODY_SHORT, open, high, low, close, i - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_UPSIDEGAP2CROWS
pub fn cdl_upsidegap2crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    let lookback = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    if len <= lookback { return Ok(output); }

    let mut body_long_sum = 0.0;
    let mut body_short_sum = 0.0;
    let start = lookback;
    for i in (start - 2 - BODY_LONG.avg_period)..(start - 2) { body_long_sum += cr(BODY_LONG, open, high, low, close, i); }
    for i in (start - 1 - BODY_SHORT.avg_period)..(start - 1) { body_short_sum += cr(BODY_SHORT, open, high, low, close, i); }

    for i in start..len {
        if candle_color(open[i-2], close[i-2]) == 1
            && real_body(open[i-2], close[i-2]) > ca(BODY_LONG, body_long_sum, open, high, low, close, i-2)
            // 2nd: short black, gap up
            && candle_color(open[i-1], close[i-1]) == -1
            && real_body(open[i-1], close[i-1]) <= ca(BODY_SHORT, body_short_sum, open, high, low, close, i-1)
            && real_body_gap_up(open, close, i-1, i-2)
            // 3rd: black, engulfs 2nd body, close above 1st close
            && candle_color(open[i], close[i]) == -1
            && open[i] > open[i-1]
            && close[i] < close[i-1]
            && close[i] > close[i-2]
        {
            output[i] = -100;
        }
        body_long_sum += cr(BODY_LONG, open, high, low, close, i-2) - cr(BODY_LONG, open, high, low, close, i - 2 - BODY_LONG.avg_period);
        body_short_sum += cr(BODY_SHORT, open, high, low, close, i-1) - cr(BODY_SHORT, open, high, low, close, i - 1 - BODY_SHORT.avg_period);
    }
    Ok(output)
}

/// CDL_XSIDEGAP3METHODS
pub fn cdl_xsidegap3methods(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<i32>> {
    let len = validate_ohlc(open, high, low, close)?;
    let mut output = vec![0i32; len];
    // lookback = 2
    if len < 3 { return Ok(output); }

    for i in 2..len {
        let c2 = candle_color(open[i-2], close[i-2]);
        let c1 = candle_color(open[i-1], close[i-1]);
        let c0 = candle_color(open[i], close[i]);

        if c2 == c1 && c0 != c2 {
            // 3rd opens within 2nd body, closes within 1st body
            let opens_within = open[i] > open[i-1].min(close[i-1]) && open[i] < open[i-1].max(close[i-1]);
            let closes_within = close[i] > open[i-2].min(close[i-2]) && close[i] < open[i-2].max(close[i-2]);

            if opens_within && closes_within {
                // Upside gap
                if c2 == 1 && real_body_gap_up(open, close, i-1, i-2) {
                    output[i] = 100;
                }
                // Downside gap
                else if c2 == -1 && real_body_gap_down(open, close, i-1, i-2) {
                    output[i] = -100;
                }
            }
        }
    }
    Ok(output)
}
