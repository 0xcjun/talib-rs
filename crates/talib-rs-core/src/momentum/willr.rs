use crate::error::{TaError, TaResult};
use crate::sliding_window::{sliding_max, sliding_min};

/// Williams %R — O(n) 单调队列算法
///
/// WILLR = -100 * (highest_high - close) / (highest_high - lowest_low)
/// lookback = timeperiod - 1
pub fn willr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let lookback = timeperiod - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let max_results = sliding_max(high, timeperiod);
    let min_results = sliding_min(low, timeperiod);

    let mut output = vec![f64::NAN; len];
    for (j, i) in (lookback..len).enumerate() {
        let (hh, _) = max_results[j];
        let (ll, _) = min_results[j];
        let range = hh - ll;
        output[i] = if range > 0.0 {
            -100.0 * (hh - close[i]) / range
        } else {
            0.0
        };
    }

    Ok(output)
}
