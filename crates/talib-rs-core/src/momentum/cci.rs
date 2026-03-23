use crate::error::{TaError, TaResult};

/// Commodity Channel Index (CCI) — O(n) average + O(n*p) mean deviation
///
/// CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
/// TP = (High + Low + Close) / 3
///
/// Optimization: sliding sum for SMA (O(1) per bar) replaces C TA-Lib's
/// O(p) full-buffer rescan. Mean deviation still requires O(p) scan
/// (no known O(1) sliding algorithm for |x - mean|).
///
/// Net effect: ~50% less work per bar vs C TA-Lib / original implementation.
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

    let tp_f = timeperiod as f64;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Circular buffer for TP values
    let mut circ_buf = vec![0.0_f64; timeperiod];
    let mut circ_idx: usize = 0;

    // Initialize: fill buffer and compute initial sum
    let mut running_sum = 0.0_f64;
    for i in 0..timeperiod {
        let tp = (high[i] + low[i] + close[i]) / 3.0;
        circ_buf[i] = tp;
        running_sum += tp;
    }
    circ_idx = 0; // Will overwrite from position 0

    // First output at lookback
    {
        let last_value = circ_buf[lookback];
        let the_average = running_sum / tp_f;
        let mut mean_dev_sum = 0.0_f64;
        for j in 0..timeperiod {
            mean_dev_sum += (circ_buf[j] - the_average).abs();
        }
        let mean_dev = mean_dev_sum / tp_f;
        output[lookback] = if mean_dev > 0.0 {
            (last_value - the_average) / (0.015 * mean_dev)
        } else {
            0.0
        };
    }

    // Main loop: sliding sum for average, full scan for mean deviation
    circ_idx = 0;
    for i in (lookback + 1)..len {
        let new_tp = (high[i] + low[i] + close[i]) / 3.0;

        // Sliding sum: O(1) average update
        running_sum += new_tp - circ_buf[circ_idx];
        circ_buf[circ_idx] = new_tp;

        let the_average = running_sum / tp_f;

        // Mean deviation: O(p) scan (unavoidable)
        let mut mean_dev_sum = 0.0_f64;
        for j in 0..timeperiod {
            mean_dev_sum += (circ_buf[j] - the_average).abs();
        }
        let mean_dev = mean_dev_sum / tp_f;

        output[i] = if mean_dev > 0.0 {
            (new_tp - the_average) / (0.015 * mean_dev)
        } else {
            0.0
        };

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
