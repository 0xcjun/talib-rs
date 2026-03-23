use crate::error::{TaError, TaResult};

/// Average Directional Index (ADX)
///
/// lookback = 2 * timeperiod - 1
///
/// Computes TR, +DM, -DM inline (no intermediate Vec allocations),
/// applies Wilder smoothing for DI, then Wilder smoothing for ADX.
pub fn adx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
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
    let lookback = 2 * timeperiod - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let pf = timeperiod as f64;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Phase 1: Compute raw TR, +DM, -DM and seed Wilder sums.
    // Seed range: bars 1..timeperiod-1 (i.e. period-2 bars for the seed sum).
    let mut sum_tr: f64 = 0.0;
    let mut sum_pdm: f64 = 0.0;
    let mut sum_mdm: f64 = 0.0;

    for i in 1..timeperiod {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        sum_tr += hl.max(hc).max(lc);

        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        if up > down && up > 0.0 {
            sum_pdm += up;
        } else if down > up && down > 0.0 {
            sum_mdm += down;
        }
    }

    // Phase 2: Wilder smooth TR/+DM/-DM from timeperiod..len, compute DX values.
    // We need DX values starting at index `timeperiod` to seed ADX.
    // ADX seed = SMA of first `timeperiod` DX values (indices timeperiod..2*timeperiod-1).
    let dx_start = timeperiod;
    let adx_start = dx_start + timeperiod - 1; // = 2*timeperiod - 1

    let mut dx_sum_for_adx_seed: f64 = 0.0;
    let mut dx_count_for_adx_seed: usize = 0;
    let mut prev_adx: f64 = 0.0;

    for i in timeperiod..len {
        // Compute TR, +DM, -DM for bar i
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        let tr_i = hl.max(hc).max(lc);

        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        let pdm_i = if up > down && up > 0.0 { up } else { 0.0 };
        let mdm_i = if down > up && down > 0.0 { down } else { 0.0 };

        // Wilder smoothing
        sum_tr = sum_tr - sum_tr / pf + tr_i;
        sum_pdm = sum_pdm - sum_pdm / pf + pdm_i;
        sum_mdm = sum_mdm - sum_mdm / pf + mdm_i;

        // Compute DX
        let dx_val = if sum_tr > 0.0 {
            let pdi = 100.0 * sum_pdm / sum_tr;
            let mdi = 100.0 * sum_mdm / sum_tr;
            let sum_di = pdi + mdi;
            if sum_di > 0.0 {
                100.0 * (pdi - mdi).abs() / sum_di
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        if i < adx_start {
            // Accumulate DX for ADX seed
            if !dx_val.is_nan() {
                dx_sum_for_adx_seed += dx_val;
                dx_count_for_adx_seed += 1;
            }
        } else if i == adx_start {
            // Include this DX in seed, then compute initial ADX = SMA(DX)
            if !dx_val.is_nan() {
                dx_sum_for_adx_seed += dx_val;
                dx_count_for_adx_seed += 1;
            }
            if dx_count_for_adx_seed > 0 {
                prev_adx = dx_sum_for_adx_seed / dx_count_for_adx_seed as f64;
                output[i] = prev_adx;
            }
        } else {
            // Wilder smooth ADX
            if !dx_val.is_nan() {
                prev_adx = (prev_adx * (pf - 1.0) + dx_val) / pf;
                output[i] = prev_adx;
            }
        }
    }

    Ok(output)
}

/// Average Directional Movement Index Rating (ADXR)
///
/// ADXR = (ADX_today + ADX_period_ago) / 2
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let adx_values = adx(high, low, close, timeperiod)?;
    let len = adx_values.len();
    let mut output = vec![0.0_f64; len];
    // ADXR lookback = 3*timeperiod - 2 (ADX lookback + timeperiod - 1)
    let adxr_lookback = 3 * timeperiod - 2;
    output[..adxr_lookback.min(len)].fill(f64::NAN);

    for i in 0..len {
        if !adx_values[i].is_nan() && i >= timeperiod {
            if !adx_values[i - timeperiod + 1].is_nan() {
                output[i] = (adx_values[i] + adx_values[i - timeperiod + 1]) / 2.0;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx_basic() {
        let high: Vec<f64> = (0..50)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0 + 2.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0 - 2.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let result = adx(&high, &low, &close, 14).unwrap();
        // 找到第一个非 NaN 值
        let first_valid = result.iter().position(|v| !v.is_nan()).unwrap();
        assert!(first_valid > 0, "should have NaN lookback prefix");
        assert!(!result[first_valid].is_nan());
    }
}
