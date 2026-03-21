use crate::error::{TaError, TaResult};

/// MIDPOINT — 周期内最高与最低值的中点
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
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let mut hi = f64::NEG_INFINITY;
        let mut lo = f64::INFINITY;
        for j in start..=i {
            if input[j] > hi {
                hi = input[j];
            }
            if input[j] < lo {
                lo = input[j];
            }
        }
        output[i] = (hi + lo) / 2.0;
    }

    Ok(output)
}

/// MIDPRICE — 周期内 high 最高与 low 最低的中点
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
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        for j in start..=i {
            if high[j] > hh {
                hh = high[j];
            }
            if low[j] < ll {
                ll = low[j];
            }
        }
        output[i] = (hh + ll) / 2.0;
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
        // midpoint[2] = (max(1,5,3) + min(1,5,3)) / 2 = (5+1)/2 = 3
        assert!((result[2] - 3.0).abs() < 1e-10);
    }
}
