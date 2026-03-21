use super::true_range_array;
use crate::error::{TaError, TaResult};

/// True Range (TRANGE)
pub fn trange(high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if len < 2 {
        return Err(TaError::InsufficientData { need: 2, got: len });
    }

    let tr = true_range_array(high, low, close);
    let mut output = vec![f64::NAN; len];
    // lookback = 1 (需要前一天的 close)
    for i in 1..len {
        output[i] = tr[i];
    }
    Ok(output)
}

/// Average True Range (ATR)
///
/// 使用 Wilder 平滑，lookback = timeperiod
pub fn atr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if timeperiod < 1 || len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let tr = true_range_array(high, low, close);
    let mut output = vec![f64::NAN; len];

    // 初始 ATR = SMA(TR)
    let mut sum: f64 = tr[1..=timeperiod].iter().sum();
    let mut prev_atr = sum / timeperiod as f64;
    output[timeperiod] = prev_atr;

    // Wilder 平滑
    let pf = timeperiod as f64;
    for i in (timeperiod + 1)..len {
        prev_atr = (prev_atr * (pf - 1.0) + tr[i]) / pf;
        output[i] = prev_atr;
    }

    Ok(output)
}

/// Normalized Average True Range (NATR)
///
/// NATR = (ATR / Close) * 100
pub fn natr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let atr_values = atr(high, low, close, timeperiod)?;
    let len = close.len();
    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if !atr_values[i].is_nan() && close[i] != 0.0 {
            output[i] = (atr_values[i] / close[i]) * 100.0;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_basic() {
        let high: Vec<f64> = (0..30)
            .map(|i| 52.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let low: Vec<f64> = (0..30)
            .map(|i| 48.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let close: Vec<f64> = (0..30)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let result = atr(&high, &low, &close, 14).unwrap();
        assert!(result[13].is_nan());
        assert!(!result[14].is_nan());
        assert!(result[14] > 0.0);
    }
}
