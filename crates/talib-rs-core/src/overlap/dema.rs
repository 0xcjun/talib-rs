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
    let p = timeperiod - 1;
    let tp = timeperiod as f64;

    // Phase 1: Build EMA1 values [p..2p], accumulate SMA for EMA2 seed
    let seed1 = sum_f64(&input[..timeperiod]) / tp;
    let mut e1 = seed1;
    let mut sum2 = seed1;
    for i in timeperiod..(2 * p + 1) {
        e1 = k.mul_add(input[i] - e1, e1);
        sum2 += e1;
    }

    // Phase 2: EMA2 seeded, compute DEMA output in single fused pass
    let seed2 = sum2 / tp;
    let mut e2 = seed2;

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);
    // First output: at index lookback = 2*p
    // Need e1 at lookback — it's the current e1 after phase 1
    output[lookback] = 2.0 * e1 - e2;

    // Steady state: both EMAs cascade per bar, no intermediate Vec
    for i in (lookback + 1)..len {
        e1 = k.mul_add(input[i] - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        output[i] = 2.0 * e1 - e2;
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
