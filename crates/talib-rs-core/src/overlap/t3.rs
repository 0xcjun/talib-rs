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
    let p = timeperiod - 1;
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
        e1 = k.mul_add(input[i] - e1, e1);
        sum2 += e1;
    }

    // Phase 2: Build up EMA2 values [2p .. 3p], accumulate SMA for EMA3 seed
    let seed2 = sum2 / tp;
    let mut e2 = seed2;
    let mut sum3 = seed2;
    for i in (2 * p + 1)..(3 * p + 1) {
        let inp = input[i];
        e1 = k.mul_add(inp - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        sum3 += e2;
    }

    // Phase 3: Build up EMA3 values [3p .. 4p], accumulate SMA for EMA4 seed
    let seed3 = sum3 / tp;
    let mut e3 = seed3;
    let mut sum4 = seed3;
    for i in (3 * p + 1)..(4 * p + 1) {
        let inp = input[i];
        e1 = k.mul_add(inp - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        e3 = k.mul_add(e2 - e3, e3);
        sum4 += e3;
    }

    // Phase 4: Build up EMA4 values [4p .. 5p], accumulate SMA for EMA5 seed
    let seed4 = sum4 / tp;
    let mut e4 = seed4;
    let mut sum5 = seed4;
    for i in (4 * p + 1)..(5 * p + 1) {
        let inp = input[i];
        e1 = k.mul_add(inp - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        e3 = k.mul_add(e2 - e3, e3);
        e4 = k.mul_add(e3 - e4, e4);
        sum5 += e4;
    }

    // Phase 5: Build up EMA5 values [5p .. 6p], accumulate SMA for EMA6 seed
    let seed5 = sum5 / tp;
    let mut e5 = seed5;
    let mut sum6 = seed5;
    for i in (5 * p + 1)..(6 * p + 1) {
        let inp = input[i];
        e1 = k.mul_add(inp - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        e3 = k.mul_add(e2 - e3, e3);
        e4 = k.mul_add(e3 - e4, e4);
        e5 = k.mul_add(e4 - e5, e5);
        sum6 += e5;
    }

    // Phase 6: EMA6 seed ready, compute first output at index 6*p = lookback
    let seed6 = sum6 / tp;
    let mut e6 = seed6;
    output[lookback] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;

    // Steady state: all 6 layers cascade per bar
    for i in (lookback + 1)..len {
        e1 = k.mul_add(input[i] - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        e3 = k.mul_add(e2 - e3, e3);
        e4 = k.mul_add(e3 - e4, e4);
        e5 = k.mul_add(e4 - e5, e5);
        e6 = k.mul_add(e5 - e6, e6);
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

    /// Verify output is reasonable (exact match tested against C TA-Lib via Python alignment tests).
    #[test]
    fn test_t3_values_reasonable() {
        for period in [2, 3, 5] {
            let input: Vec<f64> = (1..=80).map(|x| (x as f64) * 1.1 + 0.3).collect();
            let result = t3(&input, period, 0.7).unwrap();
            let lookback = 6 * (period - 1);
            // NaN before lookback
            for i in 0..lookback { assert!(result[i].is_nan()); }
            // Valid after lookback
            for i in lookback..80 { assert!(!result[i].is_nan()); }
        }
    }
}
