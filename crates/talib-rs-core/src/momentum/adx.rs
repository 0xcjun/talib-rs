use crate::error::{TaError, TaResult};
use crate::volatility::true_range_array;

/// 计算 +DM (Plus Directional Movement)
fn plus_dm_raw(high: &[f64], low: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![0.0; len];
    for i in 1..len {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if up_move > down_move && up_move > 0.0 {
            result[i] = up_move;
        }
    }
    result
}

/// 计算 -DM (Minus Directional Movement)
fn minus_dm_raw(high: &[f64], low: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![0.0; len];
    for i in 1..len {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if down_move > up_move && down_move > 0.0 {
            result[i] = down_move;
        }
    }
    result
}

/// Wilder 平滑求和 (用于 DM 和 TR)
fn wilder_sum(data: &[f64], period: usize, start: usize) -> (f64, Vec<f64>) {
    let len = data.len();
    let mut result = vec![f64::NAN; len];
    let mut sum: f64 = data[start..start + period].iter().sum();
    result[start + period - 1] = sum;
    for i in (start + period)..len {
        sum = sum - sum / period as f64 + data[i];
        result[i] = sum;
    }
    (sum, result)
}

/// Average Directional Index (ADX)
///
/// lookback = 2 * timeperiod
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
    let lookback = 2 * timeperiod;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let tr = true_range_array(high, low, close);
    let pdm = plus_dm_raw(high, low);
    let mdm = minus_dm_raw(high, low);

    // Wilder 平滑 TR, +DM, -DM
    let (_, smoothed_tr) = wilder_sum(&tr, timeperiod, 1);
    let (_, smoothed_pdm) = wilder_sum(&pdm, timeperiod, 1);
    let (_, smoothed_mdm) = wilder_sum(&mdm, timeperiod, 1);

    // DX
    let dx_start = timeperiod; // smoothed values 从 timeperiod 开始有效
    let mut dx_values = vec![f64::NAN; len];
    for i in dx_start..len {
        if !smoothed_tr[i].is_nan() && smoothed_tr[i] > 0.0 {
            let pdi = 100.0 * smoothed_pdm[i] / smoothed_tr[i];
            let mdi = 100.0 * smoothed_mdm[i] / smoothed_tr[i];
            let sum = pdi + mdi;
            if sum > 0.0 {
                dx_values[i] = 100.0 * (pdi - mdi).abs() / sum;
            } else {
                dx_values[i] = 0.0;
            }
        }
    }

    // ADX = Wilder 平滑(DX)
    let mut output = vec![f64::NAN; len];
    let adx_start = dx_start + timeperiod - 1;

    if adx_start < len {
        // 初始 ADX = SMA(DX)
        let mut sum = 0.0;
        let mut count = 0;
        for i in dx_start..=adx_start.min(len - 1) {
            if !dx_values[i].is_nan() {
                sum += dx_values[i];
                count += 1;
            }
        }
        if count > 0 {
            let mut prev_adx = sum / count as f64;
            output[adx_start] = prev_adx;

            for i in (adx_start + 1)..len {
                if !dx_values[i].is_nan() {
                    prev_adx =
                        (prev_adx * (timeperiod as f64 - 1.0) + dx_values[i]) / timeperiod as f64;
                    output[i] = prev_adx;
                }
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
    let mut output = vec![f64::NAN; len];

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
