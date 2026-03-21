use crate::error::{TaError, TaResult};

/// Williams %R
///
/// WILLR = -100 * (highest_high - close) / (highest_high - lowest_low)
/// lookback = timeperiod - 1
pub fn willr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
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

    let mut output = vec![f64::NAN; len];
    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        for j in start..=i {
            if high[j] > hh { hh = high[j]; }
            if low[j] < ll { ll = low[j]; }
        }
        let range = hh - ll;
        if range > 0.0 {
            output[i] = -100.0 * (hh - close[i]) / range;
        } else {
            output[i] = 0.0;
        }
    }

    Ok(output)
}
