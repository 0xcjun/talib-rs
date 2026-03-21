use crate::error::{TaError, TaResult};

/// Commodity Channel Index (CCI)
///
/// CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
/// TP (Typical Price) = (High + Low + Close) / 3
/// lookback = timeperiod - 1
pub fn cci(
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

    // 计算 Typical Price
    let tp: Vec<f64> = (0..len)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();

    let mut output = vec![f64::NAN; len];

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let mean: f64 = tp[start..=i].iter().sum::<f64>() / timeperiod as f64;
        let mean_dev: f64 = tp[start..=i]
            .iter()
            .map(|v| (v - mean).abs())
            .sum::<f64>() / timeperiod as f64;

        if mean_dev > 0.0 {
            output[i] = (tp[i] - mean) / (0.015 * mean_dev);
        } else {
            output[i] = 0.0;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cci_basic() {
        let high: Vec<f64> = (0..30).map(|i| 52.0 + (i as f64 * 0.3).sin() * 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 48.0 + (i as f64 * 0.3).sin() * 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0).collect();
        let result = cci(&high, &low, &close, 14).unwrap();
        assert!(result[12].is_nan());
        assert!(!result[13].is_nan());
    }
}
