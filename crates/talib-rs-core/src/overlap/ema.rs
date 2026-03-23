use crate::error::{TaError, TaResult};

/// EMA 核心计算，可被其他指标复用。
///
/// 使用 TA-Lib 兼容的初始化方式：第一个 EMA 值 = SMA(前 period 个元素)。
/// 平滑因子 k = 2.0 / (period + 1)
///
/// 返回完整输出数组（前 lookback 个为 NaN）。
pub fn ema_core(input: &[f64], period: usize, k: f64) -> TaResult<Vec<f64>> {
    let len = input.len();
    if period == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    if len < period {
        return Err(TaError::InsufficientData {
            need: period,
            got: len,
        });
    }

    let mut output = vec![0.0; len];
    let lookback = period - 1;

    // 前 lookback 个为 NaN
    for i in 0..lookback {
        output[i] = f64::NAN;
    }

    // 初始值 = SMA(前 period 个元素)，与 TA-Lib 一致
    let sma_seed: f64 = crate::simd::sum_f64(&input[..period]) / period as f64;
    output[lookback] = sma_seed;

    // 递推 EMA — 标量累加器 + unsafe 避免边界检查
    let one_minus_k = 1.0 - k;
    let mut prev = sma_seed;
    for i in period..len {
        unsafe {
            let val = *input.get_unchecked(i) * k + prev * one_minus_k;
            *output.get_unchecked_mut(i) = val;
            prev = val;
        }
    }

    Ok(output)
}

/// Exponential Moving Average (EMA)
///
/// 标准 EMA，平滑因子 k = 2.0 / (timeperiod + 1)
/// lookback = timeperiod - 1
pub fn ema(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let k = 2.0 / (timeperiod as f64 + 1.0);
    ema_core(input, timeperiod, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema(&input, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // 第一个 EMA = SMA(1,2,3) = 2.0
        assert!((result[2] - 2.0).abs() < 1e-10);
        // EMA[3] = 4 * 0.5 + 2.0 * 0.5 = 3.0
        assert!((result[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_period_1() {
        // period=1 时 k=1.0，EMA 等于原值
        let input = vec![1.0, 2.0, 3.0];
        let result = ema(&input, 1).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }
}
