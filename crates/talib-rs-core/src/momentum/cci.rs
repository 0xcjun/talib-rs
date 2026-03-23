use crate::error::{TaError, TaResult};

/// Commodity Channel Index (CCI)
///
/// CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
/// TP (Typical Price) = (High + Low + Close) / 3
///
/// Faithful port of C TA-Lib ta_CCI.c: uses circular buffer and recomputes
/// average + mean deviation from scratch each bar (avoids floating point drift).
/// lookback = timeperiod - 1
pub fn cci(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
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

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Circular buffer matching C TA-Lib
    let mut circ_buf = vec![0.0_f64; timeperiod];
    let mut circ_idx: usize = 0;

    // Fill initial circular buffer
    for i in 0..timeperiod {
        circ_buf[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    circ_idx = timeperiod - 1;

    // Process from lookback onwards
    for i in lookback..len {
        // Store current TP in circular buffer
        let last_value = (high[i] + low[i] + close[i]) / 3.0;
        circ_buf[circ_idx] = last_value;

        // Recompute average from circular buffer (no running sum, matches C TA-Lib)
        let mut the_average = 0.0_f64;
        for j in 0..timeperiod {
            the_average += circ_buf[j];
        }
        the_average /= timeperiod as f64;

        // Recompute mean deviation from circular buffer
        let mut temp_real2 = 0.0_f64;
        for j in 0..timeperiod {
            temp_real2 += (circ_buf[j] - the_average).abs();
        }

        let temp_real = last_value - the_average;
        let mean_dev = temp_real2 / timeperiod as f64;

        if mean_dev > 0.0 {
            output[i] = temp_real / (0.015 * mean_dev);
        } else {
            output[i] = 0.0;
        }

        // Advance circular buffer index
        circ_idx += 1;
        if circ_idx >= timeperiod {
            circ_idx = 0;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cci_basic() {
        let high: Vec<f64> = (0..30)
            .map(|i| 52.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let low: Vec<f64> = (0..30)
            .map(|i| 48.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let close: Vec<f64> = (0..30)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let result = cci(&high, &low, &close, 14).unwrap();
        assert!(result[12].is_nan());
        assert!(!result[13].is_nan());
    }
}
