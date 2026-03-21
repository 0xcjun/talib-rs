use crate::error::{TaError, TaResult};

/// Triangular Moving Average (TRIMA)
///
/// TRIMA = SMA(SMA(input, period1), period2)
/// 其中:
///   奇数 period: period1 = period2 = (period + 1) / 2
///   偶数 period: period1 = period / 2 + 1, period2 = period / 2
/// lookback = timeperiod - 1
pub fn trima(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
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

    // TA-Lib 实现: 直接使用三角权重
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    // 构建三角权重
    let half = timeperiod / 2;
    let mut weights = Vec::with_capacity(timeperiod);
    if timeperiod % 2 == 1 {
        // 奇数: 1,2,...,half+1,...,2,1
        for i in 0..=half {
            weights.push((i + 1) as f64);
        }
        for i in (0..half).rev() {
            weights.push((i + 1) as f64);
        }
    } else {
        // 偶数: 1,2,...,half,half,...,2,1
        for i in 0..half {
            weights.push((i + 1) as f64);
        }
        for i in (0..half).rev() {
            weights.push((i + 1) as f64);
        }
    }

    let weight_sum: f64 = weights.iter().sum();

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let mut sum = 0.0;
        for (j, &w) in weights.iter().enumerate() {
            sum += input[start + j] * w;
        }
        output[i] = sum / weight_sum;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trima_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = trima(&input, 5).unwrap();
        assert!(result[3].is_nan());
        assert!(!result[4].is_nan());
    }
}
