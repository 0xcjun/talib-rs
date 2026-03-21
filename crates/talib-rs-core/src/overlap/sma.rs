use crate::error::{TaError, TaResult};

/// Simple Moving Average (SMA)
///
/// 计算给定周期的简单移动平均线。
/// lookback = timeperiod - 1
pub fn sma(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    // 初始窗口求和 (SIMD 加速)
    let mut sum: f64 = crate::simd::sum_f64(&input[..timeperiod]);
    output[lookback] = sum / timeperiod as f64;

    // 滑动窗口 O(1)
    for i in timeperiod..len {
        sum += input[i] - input[i - timeperiod];
        output[i] = sum / timeperiod as f64;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = sma(&input, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[9] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma_period_1() {
        let input = vec![1.0, 2.0, 3.0];
        let result = sma(&input, 1).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma_insufficient_data() {
        let input = vec![1.0, 2.0];
        assert!(sma(&input, 3).is_err());
    }

    #[test]
    fn test_sma_invalid_period() {
        let input = vec![1.0, 2.0, 3.0];
        assert!(sma(&input, 0).is_err());
    }
}
