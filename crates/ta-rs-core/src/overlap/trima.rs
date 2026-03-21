use crate::error::{TaError, TaResult};

/// Triangular Moving Average (TRIMA) — O(n) 双重 SMA 算法
///
/// TRIMA = SMA(SMA(input, p1), p2)，用两次滑动窗口实现。
///   奇数 period: p1 = p2 = (period + 1) / 2
///   偶数 period: p1 = period/2 + 1, p2 = period/2
/// lookback = timeperiod - 1
pub fn trima(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    if timeperiod == 1 {
        return Ok(input.to_vec());
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    // 分解为两次 SMA
    let (p1, p2) = if timeperiod % 2 == 1 {
        let h = (timeperiod + 1) / 2;
        (h, h)
    } else {
        (timeperiod / 2 + 1, timeperiod / 2)
    };

    // 第一次 SMA: O(n) 滑动窗口
    let mut sma1 = vec![f64::NAN; len];
    let lookback1 = p1 - 1;
    let mut sum1: f64 = input[..p1].iter().sum();
    sma1[lookback1] = sum1 / p1 as f64;
    for i in p1..len {
        sum1 += input[i] - input[i - p1];
        sma1[i] = sum1 / p1 as f64;
    }

    // 第二次 SMA 在 sma1 的有效值上: O(n)
    let mut output = vec![f64::NAN; len];
    let start2 = lookback1 + p2 - 1; // = p1 - 1 + p2 - 1 = timeperiod - 1
    if start2 < len {
        let mut sum2: f64 = sma1[lookback1..(lookback1 + p2)].iter().sum();
        output[start2] = sum2 / p2 as f64;
        for i in (start2 + 1)..len {
            sum2 += sma1[i] - sma1[i - p2];
            output[i] = sum2 / p2 as f64;
        }
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

    #[test]
    fn test_trima_period_1() {
        let input = vec![1.0, 2.0, 3.0];
        let result = trima(&input, 1).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
    }
}
