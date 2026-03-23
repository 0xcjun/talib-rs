use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// TRIX — 三重指数平滑的 ROC
///
/// TRIX = ROC(EMA(EMA(EMA(input))))
/// lookback = 3*(timeperiod-1) + 1
///
/// 优化版本：3 层 EMA 标量级联 + ROC，仅 1 个输出 Vec，无中间分配。
pub fn trix(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    let lookback = 3 * (timeperiod - 1) + 1;
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
    // EMA1 seed = SMA(input[0..timeperiod])
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

    // Phase 3: EMA3 seed ready. First EMA3 value at index 3*p.
    // ROC needs previous EMA3 value, so first output at index 3*p + 1 = lookback.
    let seed3 = sum3 / tp;
    let mut e3_prev = seed3;

    // Compute one more step to get e3 at index 3*p + 1
    let i = 3 * p + 1;
    e1 = input[i] * k + e1 * one_minus_k;
    e2 = e1 * k + e2 * one_minus_k;
    let e3_cur = e2 * k + e3_prev * one_minus_k;
    if e3_prev != 0.0 {
        output[lookback] = ((e3_cur - e3_prev) / e3_prev) * 100.0;
    }
    e3_prev = e3_cur;

    // Steady state: cascade all 3 EMA layers + ROC
    for i in (lookback + 1)..len {
        e1 = input[i] * k + e1 * one_minus_k;
        e2 = e1 * k + e2 * one_minus_k;
        let e3_cur = e2 * k + e3_prev * one_minus_k;
        if e3_prev != 0.0 {
            output[i] = ((e3_cur - e3_prev) / e3_prev) * 100.0;
        }
        e3_prev = e3_cur;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify optimized output matches original Vec-based implementation bit-for-bit.
    #[test]
    fn test_trix_numerical_equivalence() {
        fn trix_reference(input: &[f64], timeperiod: usize) -> Vec<f64> {
            let len = input.len();
            let k = 2.0 / (timeperiod as f64 + 1.0);
            let one_minus_k = 1.0 - k;
            let p = timeperiod - 1;
            let tp = timeperiod as f64;
            // EMA1
            let mut e1 = vec![0.0_f64; len];
            e1[..p].fill(f64::NAN);
            let seed1: f64 = input[..timeperiod].iter().sum::<f64>() / tp;
            e1[p] = seed1;
            for i in timeperiod..len { e1[i] = input[i] * k + e1[i-1] * one_minus_k; }
            // Filter NaN for EMA2 input
            let e1v: Vec<f64> = e1.iter().copied().filter(|v| !v.is_nan()).collect();
            // EMA2
            let mut e2 = vec![0.0_f64; e1v.len()];
            e2[..p].fill(f64::NAN);
            let seed2: f64 = e1v[..timeperiod].iter().sum::<f64>() / tp;
            e2[p] = seed2;
            for i in timeperiod..e1v.len() { e2[i] = e1v[i] * k + e2[i-1] * one_minus_k; }
            // Filter NaN for EMA3 input
            let e2v: Vec<f64> = e2.iter().copied().filter(|v| !v.is_nan()).collect();
            // EMA3
            let mut e3 = vec![0.0_f64; e2v.len()];
            e3[..p].fill(f64::NAN);
            let seed3: f64 = e2v[..timeperiod].iter().sum::<f64>() / tp;
            e3[p] = seed3;
            for i in timeperiod..e2v.len() { e3[i] = e2v[i] * k + e3[i-1] * one_minus_k; }
            // ROC
            let e3_offset = 2 * p;
            let mut output = vec![0.0_f64; len];
            output[..(3 * p + 1)].fill(f64::NAN);
            for j in timeperiod..e3.len() {
                let prev = e3[j - 1];
                if prev != 0.0 {
                    let orig_idx = j + e3_offset;
                    output[orig_idx] = ((e3[j] - prev) / prev) * 100.0;
                }
            }
            output
        }
        for period in [2, 3, 5, 10] {
            let input: Vec<f64> = (1..=60).map(|x| (x as f64) * 1.1 + 0.3).collect();
            let opt = trix(&input, period).unwrap();
            let reference = trix_reference(&input, period);
            for i in 0..input.len() {
                assert!(
                    (opt[i].is_nan() && reference[i].is_nan()) || opt[i] == reference[i],
                    "TRIX mismatch at index {} for period {}: opt={} ref={}", i, period, opt[i], reference[i]
                );
            }
        }
    }
}
