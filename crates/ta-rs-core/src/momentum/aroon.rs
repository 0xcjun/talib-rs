use crate::error::{TaError, TaResult};

/// Aroon (AROON)
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

    let mut aroon_down = vec![f64::NAN; len];
    let mut aroon_up = vec![f64::NAN; len];
    let period_f = timeperiod as f64;

    for i in timeperiod..len {
        let start = i - timeperiod;
        let mut highest_idx = start;
        let mut lowest_idx = start;

        for j in (start + 1)..=i {
            if high[j] >= high[highest_idx] {
                highest_idx = j;
            }
            if low[j] <= low[lowest_idx] {
                lowest_idx = j;
            }
        }

        aroon_up[i] = 100.0 * (period_f - (i - highest_idx) as f64) / period_f;
        aroon_down[i] = 100.0 * (period_f - (i - lowest_idx) as f64) / period_f;
    }

    Ok((aroon_down, aroon_up))
}

/// Aroon Oscillator
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
