use crate::error::{TaError, TaResult};
use crate::overlap::ema::ema_core;

/// Double Exponential Moving Average (DEMA)
///
/// DEMA = 2 * EMA(input) - EMA(EMA(input))
/// lookback = 2 * (timeperiod - 1)
pub fn dema(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2 for DEMA",
        });
    }
    let len = input.len();
    let lookback = 2 * (timeperiod - 1);
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let k = 2.0 / (timeperiod as f64 + 1.0);

    // 第一次 EMA
    let ema1 = ema_core(input, timeperiod, k)?;

    // 收集非 NaN 值，进行第二次 EMA
    let ema1_valid: Vec<f64> = ema1.iter().copied().filter(|v| !v.is_nan()).collect();
    let ema2_valid = ema_core(&ema1_valid, timeperiod, k)?;

    // 构建输出: DEMA = 2*EMA1 - EMA2
    let mut output = vec![f64::NAN; len];
    let ema2_start = timeperiod - 1; // ema2_valid 中第一个非 NaN 的位置
    let output_start = lookback;

    for i in 0..(len - output_start) {
        let e1 = ema1[output_start + i]; // ema1[lookback + i]
        let e2_idx = ema2_start + i;
        if e2_idx < ema2_valid.len() {
            let e2 = ema2_valid[e2_idx];
            if !e2.is_nan() {
                output[output_start + i] = 2.0 * e1 - e2;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dema_basic() {
        let input: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = dema(&input, 5).unwrap();
        // 前 8 个应为 NaN (lookback = 2*(5-1) = 8)
        assert!(result[7].is_nan());
        assert!(!result[8].is_nan());
    }
}
