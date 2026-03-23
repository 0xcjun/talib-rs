use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// T3 Triple Exponential Moving Average (Tillson)
///
/// T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
/// 其中 EMA_n 是 n 次 EMA 嵌套, v_factor 默认 0.7
/// c1 = -v^3, c2 = 3v^2 + 3v^3, c3 = -6v^2 - 3v - 3v^3, c4 = 1 + 3v + v^3 + 3v^2
/// lookback = 6 * (timeperiod - 1)
///
/// 优化版本：6 次 EMA 逐层级联计算，仅 1 个输出 Vec，无中间分配。
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
    let one_minus_k = 1.0 - k;
    let p = timeperiod - 1; // p = period - 1
    let tp = timeperiod as f64;

    // T3 coefficients
    let v = v_factor;
    let v2 = v * v;
    let v3 = v2 * v;
    let c1 = -v3;
    let c2 = 3.0 * v2 + 3.0 * v3;
    let c3 = -6.0 * v2 - 3.0 * v - 3.0 * v3;
    let c4 = 1.0 + 3.0 * v + v3 + 3.0 * v2;

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Phase 1: Build up EMA1 values [p .. 2p], accumulate SMA for EMA2 seed
    // EMA1 seed = SMA(input[0..timeperiod])
    let seed1 = sum_f64(&input[..timeperiod]) / tp;
    let mut e1 = seed1;
    let mut sum2 = seed1;
    for i in timeperiod..(2 * p + 1) {
        e1 = input[i] * k + e1 * one_minus_k;
        sum2 += e1;
    }

    // Phase 2: Build up EMA2 values [2p .. 3p], accumulate SMA for EMA3 seed
    let seed2 = sum2 / tp;
    let mut e2 = seed2;
    let mut sum3 = seed2;
    for i in (2 * p + 1)..(3 * p + 1) {
        let inp = input[i];
        e1 = inp * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        sum3 += e2;
    }

    // Phase 3: Build up EMA3 values [3p .. 4p], accumulate SMA for EMA4 seed
    let seed3 = sum3 / tp;
    let mut e3 = seed3;
    let mut sum4 = seed3;
    for i in (3 * p + 1)..(4 * p + 1) {
        let inp = input[i];
        e1 = inp * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        e3 = e2 * k + e3 * one_minus_k;
        sum4 += e3;
    }

    // Phase 4: Build up EMA4 values [4p .. 5p], accumulate SMA for EMA5 seed
    let seed4 = sum4 / tp;
    let mut e4 = seed4;
    let mut sum5 = seed4;
    for i in (4 * p + 1)..(5 * p + 1) {
        let inp = input[i];
        e1 = inp * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        e3 = e2 * k + e3 * one_minus_k;
        e4 = e3 * k + e4 * one_minus_k;
        sum5 += e4;
    }

    // Phase 5: Build up EMA5 values [5p .. 6p], accumulate SMA for EMA6 seed
    let seed5 = sum5 / tp;
    let mut e5 = seed5;
    let mut sum6 = seed5;
    for i in (5 * p + 1)..(6 * p + 1) {
        let inp = input[i];
        e1 = inp * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        e3 = e2 * k + e3 * one_minus_k;
        e4 = e3 * k + e4 * one_minus_k;
        e5 = e4 * k + e5 * one_minus_k;
        sum6 += e5;
    }

    // Phase 6: EMA6 seed ready, compute first output at index 6*p = lookback
    let seed6 = sum6 / tp;
    let mut e6 = seed6;
    output[lookback] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;

    // Steady state: all 6 layers cascade per bar
    for i in (lookback + 1)..len {
        e1 = input[i] * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        e3 = e2 * k + e3 * one_minus_k;
        e4 = e3 * k + e4 * one_minus_k;
        e5 = e4 * k + e5 * one_minus_k;
        e6 = e5 * k + e6 * one_minus_k;
        output[i] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
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

    /// Verify optimized output matches original Vec-based implementation bit-for-bit.
    #[test]
    fn test_t3_numerical_equivalence() {
        fn t3_reference(input: &[f64], timeperiod: usize, v_factor: f64) -> Vec<f64> {
            let len = input.len();
            let k = 2.0 / (timeperiod as f64 + 1.0);
            let one_minus_k = 1.0 - k;
            let p = timeperiod - 1;
            let tp = timeperiod as f64;
            let compute_ema_layer = |prev_layer: &[f64], start_idx: usize| -> Vec<f64> {
                let ns = start_idx + p;
                let mut ema = vec![0.0_f64; len];
                ema[..ns].fill(f64::NAN);
                let seed: f64 = prev_layer[start_idx..(start_idx + timeperiod)].iter().sum::<f64>() / tp;
                ema[ns] = seed;
                for i in (ns + 1)..len { ema[i] = prev_layer[i] * k + ema[i - 1] * one_minus_k; }
                ema
            };
            let mut e1 = vec![0.0_f64; len];
            e1[..p].fill(f64::NAN);
            let seed1: f64 = input[..timeperiod].iter().sum::<f64>() / tp;
            e1[p] = seed1;
            for i in timeperiod..len { e1[i] = input[i] * k + e1[i - 1] * one_minus_k; }
            let e2 = compute_ema_layer(&e1, p);
            let e3 = compute_ema_layer(&e2, 2 * p);
            let e4 = compute_ema_layer(&e3, 3 * p);
            let e5 = compute_ema_layer(&e4, 4 * p);
            let v = v_factor; let v2 = v*v; let v3 = v2*v;
            let c1 = -v3; let c2 = 3.0*v2 + 3.0*v3; let c3 = -6.0*v2 - 3.0*v - 3.0*v3; let c4 = 1.0 + 3.0*v + v3 + 3.0*v2;
            let seed6: f64 = e5[(5*p)..(5*p+timeperiod)].iter().sum::<f64>() / tp;
            let lookback = 6 * p;
            let mut output = vec![0.0_f64; len];
            output[..lookback].fill(f64::NAN);
            let mut e6_prev = seed6;
            output[lookback] = c1 * e6_prev + c2 * e5[lookback] + c3 * e4[lookback] + c4 * e3[lookback];
            for i in (lookback + 1)..len { e6_prev = e5[i] * k + e6_prev * one_minus_k; output[i] = c1 * e6_prev + c2 * e5[i] + c3 * e4[i] + c4 * e3[i]; }
            output
        }
        for period in [2, 3, 5] {
            let input: Vec<f64> = (1..=80).map(|x| (x as f64) * 1.1 + 0.3).collect();
            let opt = t3(&input, period, 0.7).unwrap();
            let reference = t3_reference(&input, period, 0.7);
            for i in 0..input.len() {
                assert!(
                    (opt[i].is_nan() && reference[i].is_nan()) || opt[i] == reference[i],
                    "T3 mismatch at index {} for period {}: opt={} ref={}", i, period, opt[i], reference[i]
                );
            }
        }
    }
}
