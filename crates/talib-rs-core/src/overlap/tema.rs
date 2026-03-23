use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// Triple Exponential Moving Average (TEMA)
///
/// TEMA = 3*EMA1 - 3*EMA2 + EMA3
/// 其中 EMA1 = EMA(input), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
/// lookback = 3 * (timeperiod - 1)
///
/// 优化版本：三层 EMA 标量级联，仅 1 个输出 Vec，无中间分配。
/// EMA formulation: k*(x - prev) + prev — matches C TA-Lib, shorter critical path.
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
    let p = timeperiod - 1;
    let tp = timeperiod as f64;

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Phase 1: Build EMA1, indices [p .. 2p]. Accumulate SMA for EMA2 seed.
    let seed1 = sum_f64(&input[..timeperiod]) / tp;
    let mut e1 = seed1;
    let mut sum2 = seed1;
    for i in timeperiod..(2 * p + 1) {
        e1 = k.mul_add(input[i] - e1, e1);
        sum2 += e1;
    }

    // Phase 2: Build EMA2, indices [2p .. 3p]. Accumulate SMA for EMA3 seed.
    let seed2 = sum2 / tp;
    let mut e2 = seed2;
    let mut sum3 = seed2;
    for i in (2 * p + 1)..(3 * p + 1) {
        e1 = k.mul_add(input[i] - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        sum3 += e2;
    }

    // Phase 3: EMA3 seed ready at index 3*p = lookback.
    let seed3 = sum3 / tp;
    let mut e3 = seed3;
    output[lookback] = 3.0 * e1 - 3.0 * e2 + e3;

    // Steady state: cascade all 3 EMA layers
    for i in (lookback + 1)..len {
        e1 = k.mul_add(input[i] - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        e3 = k.mul_add(e2 - e3, e3);
        output[i] = 3.0 * e1 - 3.0 * e2 + e3;
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
        assert!(result[11].is_nan());
        assert!(!result[12].is_nan());
    }
}
