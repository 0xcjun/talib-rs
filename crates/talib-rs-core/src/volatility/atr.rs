use super::true_range_array;
use crate::error::{TaError, TaResult};

/// True Range (TRANGE) — zip-based for auto-vectorization
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

    let mut output = vec![0.0_f64; len];
    output[0] = f64::NAN;
    for (((&h, &l), &pc), out) in high[1..]
        .iter()
        .zip(low[1..].iter())
        .zip(close[..len - 1].iter())
        .zip(output[1..].iter_mut())
    {
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        *out = hl.max(hc).max(lc);
    }
    Ok(output)
}

/// Average True Range (ATR)
///
/// Wilder smoothing, lookback = timeperiod
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
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    let sum: f64 = tr[1..=timeperiod].iter().sum();
    let mut prev_atr = sum / timeperiod as f64;
    output[timeperiod] = prev_atr;

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

    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    // Compute initial ATR = SMA of TR for bars 1..=timeperiod
    let mut sum_tr = 0.0_f64;
    for i in 1..=timeperiod {
        let h = high[i];
        let l = low[i];
        let pc = close[i - 1];
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        sum_tr += hl.max(hc).max(lc);
    }

    let pf = timeperiod as f64;
    let mut prev_atr = sum_tr / pf;
    let c = close[timeperiod];
    if c != 0.0 {
        output[timeperiod] = (prev_atr / c) * 100.0;
    }

    // Wilder smoothing + NATR in single pass
    for i in (timeperiod + 1)..len {
        let h = high[i];
        let l = low[i];
        let pc = close[i - 1];
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        let tr_i = hl.max(hc).max(lc);

        prev_atr = (prev_atr * (pf - 1.0) + tr_i) / pf;

        let c = close[i];
        if c != 0.0 {
            output[i] = (prev_atr / c) * 100.0;
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
