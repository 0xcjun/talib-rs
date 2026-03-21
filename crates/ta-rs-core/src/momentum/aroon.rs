use crate::error::{TaError, TaResult};
use crate::sliding_window::{sliding_max, sliding_min};

/// Aroon (AROON) — O(n) 单调队列算法
///
/// 返回 (aroon_down, aroon_up)
/// Aroon Up = 100 * (period - bars since highest high) / period
/// Aroon Down = 100 * (period - bars since lowest low) / period
/// lookback = timeperiod
pub fn aroon(
    high: &[f64],
    low: &[f64],
    timeperiod: usize,
) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let period_f = timeperiod as f64;
    let window = timeperiod + 1; // AROON 的窗口包含当前 bar

    // O(n) 单调队列找每个窗口的最大/最小索引
    let max_results = sliding_max(high, window);
    let min_results = sliding_min(low, window);

    let mut aroon_down = vec![f64::NAN; len];
    let mut aroon_up = vec![f64::NAN; len];

    for (j, i) in (timeperiod..len).enumerate() {
        let (_, highest_idx) = max_results[j];
        let (_, lowest_idx) = min_results[j];
        aroon_up[i] = 100.0 * (period_f - (i - highest_idx) as f64) / period_f;
        aroon_down[i] = 100.0 * (period_f - (i - lowest_idx) as f64) / period_f;
    }

    Ok((aroon_down, aroon_up))
}

/// Aroon Oscillator — O(n)
///
/// AROONOSC = Aroon Up - Aroon Down
pub fn aroon_osc(
    high: &[f64],
    low: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let (aroon_down, aroon_up) = aroon(high, low, timeperiod)?;
    let len = high.len();
    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if !aroon_up[i].is_nan() && !aroon_down[i].is_nan() {
            output[i] = aroon_up[i] - aroon_down[i];
        }
    }
    Ok(output)
}
