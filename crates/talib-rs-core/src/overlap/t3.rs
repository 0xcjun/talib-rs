use crate::error::{TaError, TaResult};
use crate::simd::sum_f64;

/// T3 Triple Exponential Moving Average (Tillson)
///
/// T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
/// 其中 EMA_n 是 n 次 EMA 嵌套, v_factor 默认 0.7
/// c1 = -v^3, c2 = 3v^2 + 3v^3, c3 = -6v^2 - 3v - 3v^3, c4 = 1 + 3v + v^3 + 3v^2
/// lookback = 6 * (timeperiod - 1)
///
/// 优化版本：6 次 EMA 逐层计算，无需中间 Vec 分配（filter NaN）。
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
    let p = timeperiod - 1;

    // 辅助：计算一层 EMA，从 start_idx 开始（前一层有效值起始处）
    // 返回完整长度数组，仅 [start_idx + p ..] 有效
    let compute_ema_layer = |prev_layer: &[f64], start_idx: usize| -> Vec<f64> {
        let mut ema = vec![f64::NAN; len];
        let seed = sum_f64(&prev_layer[start_idx..(start_idx + timeperiod)]) / timeperiod as f64;
        let new_start = start_idx + p;
        ema[new_start] = seed;
        for i in (new_start + 1)..len {
            ema[i] = prev_layer[i] * k + ema[i - 1] * one_minus_k;
        }
        ema
    };

    // EMA1: 对 input 做 EMA，有效值从 index p 开始
    let mut e1 = vec![f64::NAN; len];
    let seed1 = sum_f64(&input[..timeperiod]) / timeperiod as f64;
    e1[p] = seed1;
    for i in timeperiod..len {
        e1[i] = input[i] * k + e1[i - 1] * one_minus_k;
    }

    // EMA2 ~ EMA5: 每层从前一层有效值起始处开始
    let e2 = compute_ema_layer(&e1, p); // 有效从 2p
    let e3 = compute_ema_layer(&e2, 2 * p); // 有效从 3p
    let e4 = compute_ema_layer(&e3, 3 * p); // 有效从 4p
    let e5 = compute_ema_layer(&e4, 4 * p); // 有效从 5p

    // EMA6: 只需要标量跟踪（最后一层），不需要完整数组
    let seed6 = sum_f64(&e5[(5 * p)..(5 * p + timeperiod)]) / timeperiod as f64;

    // T3 系数
    let v = v_factor;
    let v2 = v * v;
    let v3 = v2 * v;
    let c1 = -v3;
    let c2 = 3.0 * v2 + 3.0 * v3;
    let c3 = -6.0 * v2 - 3.0 * v - 3.0 * v3;
    let c4 = 1.0 + 3.0 * v + v3 + 3.0 * v2;

    let mut output = vec![f64::NAN; len];
    let mut e6_prev = seed6;
    // 第一个输出在 index = lookback = 6*p
    output[lookback] = c1 * e6_prev + c2 * e5[lookback] + c3 * e4[lookback] + c4 * e3[lookback];

    for i in (lookback + 1)..len {
        e6_prev = e5[i] * k + e6_prev * one_minus_k;
        output[i] = c1 * e6_prev + c2 * e5[i] + c3 * e4[i] + c4 * e3[i];
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
}
