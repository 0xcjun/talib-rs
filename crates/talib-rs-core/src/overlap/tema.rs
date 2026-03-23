use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// Triple Exponential Moving Average (TEMA)
///
/// TEMA = 3*EMA1 - 3*EMA2 + EMA3
/// 其中 EMA1 = EMA(input), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
/// lookback = 3 * (timeperiod - 1)
///
/// 优化版本：三层 EMA 标量级联，仅 1 个输出 Vec，无中间分配。
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
    let tp = timeperiod as f64;

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Phase 1: Build EMA1, indices [p .. 2p]. Accumulate SMA for EMA2 seed.
    let seed1 = sum_f64(&input[..timeperiod]) / tp;
    let mut e1 = seed1;
    let mut sum2 = seed1;
    for i in timeperiod..(2 * p + 1) {
        e1 = input[i] * k + e1 * one_minus_k;
        sum2 += e1;
    }

    // Phase 2: Build EMA2, indices [2p .. 3p]. Accumulate SMA for EMA3 seed.
    let seed2 = sum2 / tp;
    let mut e2 = seed2;
    let mut sum3 = seed2;
    for i in (2 * p + 1)..(3 * p + 1) {
        e1 = input[i] * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        sum3 += e2;
    }

    // Phase 3: EMA3 seed ready at index 3*p = lookback.
    let seed3 = sum3 / tp;
    let mut e3 = seed3;
    // First output: TEMA = 3*e1 - 3*e2 + e3
    output[lookback] = 3.0 * e1 - 3.0 * e2 + e3;

    // Steady state: cascade all 3 EMA layers
    for i in (lookback + 1)..len {
        e1 = input[i] * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        e3 = e2 * k + e3 * one_minus_k;
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
        // lookback = 3*(5-1) = 12
        assert!(result[11].is_nan());
        assert!(!result[12].is_nan());
    }

    /// Verify optimized output matches original Vec-based implementation bit-for-bit.
    #[test]
    fn test_tema_numerical_equivalence() {
        fn tema_reference(input: &[f64], timeperiod: usize) -> Vec<f64> {
            let len = input.len();
            let k = 2.0 / (timeperiod as f64 + 1.0);
            let one_minus_k = 1.0 - k;
            let p = timeperiod - 1;
            let mut ema1 = vec![0.0_f64; len];
            ema1[..p].fill(f64::NAN);
            let seed1: f64 = input[..timeperiod].iter().sum::<f64>() / timeperiod as f64;
            ema1[p] = seed1;
            for i in timeperiod..len { ema1[i] = input[i] * k + ema1[i-1] * one_minus_k; }
            let mut ema2 = vec![0.0_f64; len];
            ema2[..(2*p)].fill(f64::NAN);
            let seed2: f64 = ema1[p..(2*p+1)].iter().sum::<f64>() / timeperiod as f64;
            ema2[2*p] = seed2;
            for i in (2*p+1)..len { ema2[i] = ema1[i] * k + ema2[i-1] * one_minus_k; }
            let seed3: f64 = ema2[(2*p)..(3*p+1)].iter().sum::<f64>() / timeperiod as f64;
            let mut output = vec![0.0_f64; len];
            output[..(3*p)].fill(f64::NAN);
            let mut e3 = seed3;
            let lookback = 3 * p;
            output[lookback] = 3.0 * ema1[lookback] - 3.0 * ema2[lookback] + e3;
            for i in (lookback+1)..len { e3 = ema2[i] * k + e3 * one_minus_k; output[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + e3; }
            output
        }
        for period in [2, 3, 5, 10] {
            let input: Vec<f64> = (1..=100).map(|x| (x as f64) * 1.1 + 0.3).collect();
            let opt = tema(&input, period).unwrap();
            let reference = tema_reference(&input, period);
            for i in 0..input.len() {
                assert!(
                    (opt[i].is_nan() && reference[i].is_nan()) || opt[i] == reference[i],
                    "Mismatch at index {} for period {}: opt={} ref={}", i, period, opt[i], reference[i]
                );
            }
        }
    }
}
