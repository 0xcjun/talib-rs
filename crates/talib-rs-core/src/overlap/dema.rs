use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// Double Exponential Moving Average (DEMA)
///
/// DEMA = 2 * EMA(input) - EMA(EMA(input))
/// lookback = 2 * (timeperiod - 1)
///
/// 优化版本：两次 EMA 在原地计算，无需中间 Vec 分配。
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
    let one_minus_k = 1.0 - k;
    let p = timeperiod - 1; // EMA1 的 lookback

    // EMA1: 对 input 做 EMA，结果存入 ema1 数组
    let mut ema1 = vec![0.0_f64; len];
    let seed1 = sum_f64(&input[..timeperiod]) / timeperiod as f64;
    ema1[p] = seed1;
    let mut ema1_prev = seed1;
    for i in timeperiod..len {
        let val = input[i].mul_add(k, ema1_prev * one_minus_k);
        ema1[i] = val;
        ema1_prev = val;
    }

    // EMA2: 对 ema1[p..] 的有效值做 EMA
    let ema2_seed_start = p;
    let ema2_seed_end = 2 * p + 1;
    let seed2 = sum_f64(&ema1[ema2_seed_start..ema2_seed_end]) / timeperiod as f64;

    let mut output = vec![0.0_f64; len];
    // 前 lookback 个为 NaN
    for i in 0..lookback {
        output[i] = f64::NAN;
    }
    let mut ema2_prev = seed2;
    output[lookback] = 2.0 * ema1[lookback] - ema2_prev;

    for i in (lookback + 1)..len {
        let e1 = ema1[i];
        ema2_prev = e1.mul_add(k, ema2_prev * one_minus_k);
        output[i] = 2.0 * e1 - ema2_prev;
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
