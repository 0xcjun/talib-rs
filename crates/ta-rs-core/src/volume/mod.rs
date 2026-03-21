use crate::error::{TaError, TaResult};

/// Chaikin A/D Line (AD)
pub fn ad(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() || len != volume.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()).min(volume.len()),
        });
    }

    let mut output = vec![0.0; len];
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

/// Chaikin A/D Oscillator (ADOSC)
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
    let ad_values = ad(high, low, close, volume)?;

    use crate::overlap::ema;
    let fast = ema(&ad_values, fastperiod)?;
    let slow = ema(&ad_values, slowperiod)?;

    let len = ad_values.len();
    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if !fast[i].is_nan() && !slow[i].is_nan() {
            output[i] = fast[i] - slow[i];
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

    let mut output = vec![0.0; len];
    output[0] = volume[0];
    for i in 1..len {
        if close[i] > close[i - 1] {
            output[i] = output[i - 1] + volume[i];
        } else if close[i] < close[i - 1] {
            output[i] = output[i - 1] - volume[i];
        } else {
            output[i] = output[i - 1];
        }
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
        assert!((result[1] - 300.0).abs() < 1e-10);   // 上涨 +200
        assert!((result[2] - 150.0).abs() < 1e-10);    // 下跌 -150
    }
}
