use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// Triple Exponential Moving Average (TEMA)
///
/// TEMA = 3*EMA1 - 3*EMA2 + EMA3
/// 其中 EMA1 = EMA(input), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
/// lookback = 3 * (timeperiod - 1)
///
/// 优化版本：三次 EMA 在原地计算，无需中间 Vec 分配。
pub fn tema(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2 for TEMA",
        });
    }
    let len = input.len();
    let lookback = 3 * (timeperiod - 1);
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let k = 2.0 / (timeperiod as f64 + 1.0);
    let one_minus_k = 1.0 - k;
    let p = timeperiod - 1;

    // EMA1: 对 input 做 EMA
    let mut ema1 = vec![f64::NAN; len];
    let seed1 = sum_f64(&input[..timeperiod]) / timeperiod as f64;
    ema1[p] = seed1;
    for i in timeperiod..len {
        ema1[i] = input[i] * k + ema1[i - 1] * one_minus_k;
    }

    // EMA2: 对 ema1[p..] 做 EMA，有效值从 index p 开始
    // seed2 = SMA(ema1[p .. p + timeperiod]) = SMA(ema1[p .. 2p + 1])
    let mut ema2 = vec![f64::NAN; len];
    let seed2 = sum_f64(&ema1[p..(2 * p + 1)]) / timeperiod as f64;
    ema2[2 * p] = seed2;
    for i in (2 * p + 1)..len {
        ema2[i] = ema1[i] * k + ema2[i - 1] * one_minus_k;
    }

    // EMA3: 对 ema2[2p..] 做 EMA，有效值从 index 2p 开始
    // seed3 = SMA(ema2[2p .. 2p + timeperiod]) = SMA(ema2[2p .. 3p + 1])
    let seed3 = sum_f64(&ema2[(2 * p)..(3 * p + 1)]) / timeperiod as f64;

    let mut output = vec![f64::NAN; len];
    let mut ema3_prev = seed3;
    // 第一个 TEMA 输出在 index = lookback = 3*p
    output[lookback] = 3.0 * ema1[lookback] - 3.0 * ema2[lookback] + ema3_prev;

    for i in (lookback + 1)..len {
        ema3_prev = ema2[i] * k + ema3_prev * one_minus_k;
        output[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3_prev;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tema_basic() {
        let input: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = tema(&input, 5).unwrap();
        // lookback = 3*(5-1) = 12
        assert!(result[11].is_nan());
        assert!(!result[12].is_nan());
    }
}
