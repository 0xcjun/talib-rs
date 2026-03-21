use crate::error::{TaError, TaResult};
use crate::overlap::ema::ema_core;

/// T3 Triple Exponential Moving Average (Tillson)
///
/// T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
/// 其中 EMA_n 是 n 次 EMA 嵌套, v_factor 默认 0.7
/// c1 = -v^3, c2 = 3v^2 + 3v^3, c3 = -6v^2 - 3v - 3v^3, c4 = 1 + 3v + v^3 + 3v^2
/// lookback = 6 * (timeperiod - 1)
pub fn t3(input: &[f64], timeperiod: usize, v_factor: f64) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2 for T3",
        });
    }
    let len = input.len();
    let lookback = 6 * (timeperiod - 1);
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let k = 2.0 / (timeperiod as f64 + 1.0);

    // 6 重 EMA 嵌套
    let e1 = ema_core(input, timeperiod, k)?;
    let e1v: Vec<f64> = e1.iter().copied().filter(|v| !v.is_nan()).collect();
    let e2 = ema_core(&e1v, timeperiod, k)?;
    let e2v: Vec<f64> = e2.iter().copied().filter(|v| !v.is_nan()).collect();
    let e3 = ema_core(&e2v, timeperiod, k)?;
    let e3v: Vec<f64> = e3.iter().copied().filter(|v| !v.is_nan()).collect();
    let e4 = ema_core(&e3v, timeperiod, k)?;
    let e4v: Vec<f64> = e4.iter().copied().filter(|v| !v.is_nan()).collect();
    let e5 = ema_core(&e4v, timeperiod, k)?;
    let e5v: Vec<f64> = e5.iter().copied().filter(|v| !v.is_nan()).collect();
    let e6 = ema_core(&e5v, timeperiod, k)?;

    // T3 系数
    let v = v_factor;
    let v2 = v * v;
    let v3 = v2 * v;
    let c1 = -v3;
    let c2 = 3.0 * v2 + 3.0 * v3;
    let c3 = -6.0 * v2 - 3.0 * v - 3.0 * v3;
    let c4 = 1.0 + 3.0 * v + v3 + 3.0 * v2;

    let mut output = vec![f64::NAN; len];
    let p = timeperiod - 1;

    for i in lookback..len {
        let idx3 = i - p;
        let idx3_local = idx3 - p;
        let idx4_local = idx3_local - p;
        let idx5_local = idx4_local - p;
        let idx6_local = idx5_local - p;

        if idx3_local < e3.len()
            && idx4_local < e4.len()
            && idx5_local < e5.len()
            && idx6_local < e6.len()
        {
            let e3_val = e3[idx3_local];
            let e4_val = e4[idx4_local];
            let e5_val = e5[idx5_local];
            let e6_val = e6[idx6_local];

            if !e3_val.is_nan() && !e4_val.is_nan() && !e5_val.is_nan() && !e6_val.is_nan() {
                output[i] = c1 * e6_val + c2 * e5_val + c3 * e4_val + c4 * e3_val;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t3_basic() {
        let input: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = t3(&input, 5, 0.7).unwrap();
        // lookback = 6*(5-1) = 24
        assert!(result[23].is_nan());
        assert!(!result[24].is_nan());
    }
}
