use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// TRIX — 三重指数平滑的 ROC
///
/// TRIX = ROC(EMA(EMA(EMA(input))))
/// lookback = 3*(timeperiod-1) + 1
///
/// 优化版本：3 层 EMA 标量级联 + ROC，仅 1 个输出 Vec，无中间分配。
/// EMA formulation: k*(x - prev) + prev — matches C TA-Lib, shorter critical path.
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

    // Phase 3: EMA3 seed ready. First EMA3 value at index 3*p.
    // ROC needs previous EMA3 value, so first output at index 3*p + 1 = lookback.
    let seed3 = sum3 / tp;
    let mut e3_prev = seed3;

    // Compute one more step to get e3 at index 3*p + 1
    let i = 3 * p + 1;
    e1 = k.mul_add(input[i] - e1, e1);
    e2 = k.mul_add(e1 - e2, e2);
    let e3_cur = k.mul_add(e2 - e3_prev, e3_prev);
    if e3_prev != 0.0 {
        output[lookback] = ((e3_cur - e3_prev) / e3_prev) * 100.0;
    }
    e3_prev = e3_cur;

    // Steady state: cascade all 3 EMA layers + ROC
    for i in (lookback + 1)..len {
        e1 = k.mul_add(input[i] - e1, e1);
        e2 = k.mul_add(e1 - e2, e2);
        let e3_cur = k.mul_add(e2 - e3_prev, e3_prev);
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

    #[test]
    fn test_trix_basic() {
        let input: Vec<f64> = (1..=60).map(|x| (x as f64) * 1.1 + 0.3).collect();
        let result = trix(&input, 5).unwrap();
        let lookback = 3 * 4 + 1; // 13
        assert!(result[lookback - 1].is_nan());
        assert!(!result[lookback].is_nan());
    }
}
