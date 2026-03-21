use crate::error::{TaError, TaResult};
use crate::sliding_window::{sliding_max, sliding_min};

/// MIDPOINT — O(n) 单调队列
///
/// MIDPOINT = (highest + lowest) / 2
/// lookback = timeperiod - 1
pub fn midpoint(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }

    let max_results = sliding_max(input, timeperiod);
    let min_results = sliding_min(input, timeperiod);

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    for (j, i) in (lookback..len).enumerate() {
        output[i] = (max_results[j].0 + min_results[j].0) / 2.0;
    }
    Ok(output)
}

/// MIDPRICE — O(n) 单调队列
///
/// MIDPRICE = (highest_high + lowest_low) / 2
/// lookback = timeperiod - 1
pub fn midprice(high: &[f64], low: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch { expected: len, got: low.len() });
    }
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }

    let max_results = sliding_max(high, timeperiod);
    let min_results = sliding_min(low, timeperiod);

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    for (j, i) in (lookback..len).enumerate() {
        output[i] = (max_results[j].0 + min_results[j].0) / 2.0;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midpoint_basic() {
        let input = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let result = midpoint(&input, 3).unwrap();
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-10);
    }
}
