use crate::error::{TaError, TaResult};

/// Chaikin A/D Line (AD)
pub fn ad(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() || len != volume.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()).min(volume.len()),
        });
    }

    let mut output = vec![0.0_f64; len];
    let mut ad_val = 0.0;
    for i in 0..len {
        let hl = high[i] - low[i];
        let clv = if hl > 0.0 {
            ((close[i] - low[i]) - (high[i] - close[i])) / hl
        } else {
            0.0
        };
        ad_val += clv * volume[i];
        output[i] = ad_val;
    }

    Ok(output)
}

/// Chaikin A/D Oscillator (ADOSC) — inlined AD + dual EMA, no intermediate Vec
///
/// ADOSC = EMA(AD, fastperiod) - EMA(AD, slowperiod)
pub fn adosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fastperiod: usize,
    slowperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() || len != volume.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()).min(volume.len()),
        });
    }
    let lookback = slowperiod.max(fastperiod) - 1;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);
    let fast_k = 2.0 / (fastperiod as f64 + 1.0);
    let slow_k = 2.0 / (slowperiod as f64 + 1.0);
    // k1 variables removed — using C TA-Lib formulation: k*(x-prev)+prev

    let hl0 = high[0] - low[0];
    let ad0 = if hl0 > 0.0 {
        ((close[0] - low[0]) - (high[0] - close[0])) / hl0 * volume[0]
    } else {
        0.0
    };

    let mut ad_val = ad0;
    let mut fast_ema = ad0;
    let mut slow_ema = ad0;

    for i in 1..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];
        let hl = h - l;
        let clv = if hl > 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        ad_val += clv * v;
        fast_ema = fast_k.mul_add(ad_val - fast_ema, fast_ema);
        slow_ema = slow_k.mul_add(ad_val - slow_ema, slow_ema);
        if i >= lookback {
            output[i] = fast_ema - slow_ema;
        }
    }

    Ok(output)
}

/// On Balance Volume (OBV)
pub fn obv(close: &[f64], volume: &[f64]) -> TaResult<Vec<f64>> {
    let len = close.len();
    if len != volume.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: volume.len(),
        });
    }
    if len == 0 {
        return Ok(vec![]);
    }

    // push() is intentional: at 1M bars, push() avoids calloc COW page faults
    // that vec![0.0;n]+indexed would cause (0.98x vs 1.94x at 1M).
    // Tradeoff: 0.42x at 1K due to per-push capacity check overhead.
    let mut output = Vec::with_capacity(len);
    let mut acc = volume[0];
    output.push(acc);
    for i in 1..len {
        let c = close[i];
        let pc = close[i - 1];
        let v = volume[i];
        if c > pc {
            acc += v;
        } else if c < pc {
            acc -= v;
        }
        output.push(acc);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv_basic() {
        let close = vec![1.0, 2.0, 1.5, 3.0, 2.5];
        let volume = vec![100.0, 200.0, 150.0, 300.0, 250.0];
        let result = obv(&close, &volume).unwrap();
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert!((result[1] - 300.0).abs() < 1e-10);
        assert!((result[2] - 150.0).abs() < 1e-10);
    }
}
