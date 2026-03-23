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

/// Chaikin A/D Oscillator (ADOSC) — 内联 AD 计算，无中间 Vec
///
/// ADOSC = EMA(AD, fastperiod) - EMA(AD, slowperiod)
///
/// C TA-Lib 的 ADOSC 内联计算 EMA：以 ad[0] 为 seed，从 index 1 开始递推，
/// 输出从 max(slowperiod, fastperiod) - 1 开始。这与标准 EMA (SMA seed) 不同。
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
    let fast_k1 = 1.0 - fast_k;
    let slow_k1 = 1.0 - slow_k;

    // 内联 AD[0] 计算
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
        unsafe {
            let h = *high.get_unchecked(i);
            let l = *low.get_unchecked(i);
            let c = *close.get_unchecked(i);
            let v = *volume.get_unchecked(i);
            let hl = h - l;
            let clv = if hl > 0.0 {
                ((c - l) - (h - c)) / hl
            } else {
                0.0
            };
            ad_val += clv * v;
            fast_ema = ad_val * fast_k + fast_ema * fast_k1;
            slow_ema = ad_val * slow_k + slow_ema * slow_k1;
            if i >= lookback {
                *output.get_unchecked_mut(i) = fast_ema - slow_ema;
            }
        }
    }

    Ok(output)
}

/// On Balance Volume (OBV) — 标量累加器，无数组依赖
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
    let mut acc = unsafe { *volume.get_unchecked(0) };
    output[0] = acc;
    for i in 1..len {
        unsafe {
            let c = *close.get_unchecked(i);
            let pc = *close.get_unchecked(i - 1);
            let v = *volume.get_unchecked(i);
            if c > pc {
                acc += v;
            } else if c < pc {
                acc -= v;
            }
            *output.get_unchecked_mut(i) = acc;
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
        assert!((result[1] - 300.0).abs() < 1e-10); // 上涨 +200
        assert!((result[2] - 150.0).abs() < 1e-10); // 下跌 -150
    }
}
