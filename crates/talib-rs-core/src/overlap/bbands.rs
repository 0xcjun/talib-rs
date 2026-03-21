use crate::error::{TaError, TaResult};
use crate::ma_type::{compute_ma, MaType};

/// Bollinger Bands (BBANDS)
///
/// 返回 (upperband, middleband, lowerband)
/// middleband = MA(input, timeperiod, matype)
/// upperband = middleband + nbdevup * stddev
/// lowerband = middleband - nbdevdn * stddev
/// lookback = timeperiod - 1
pub fn bbands(
    input: &[f64],
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
    matype: MaType,
) -> TaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }

    let len = input.len();
    let ma = compute_ma(input, timeperiod, matype)?;

    let mut upper = vec![f64::NAN; len];
    let mut lower = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    for i in lookback..len {
        let ma_val = ma[i];
        if ma_val.is_nan() {
            continue;
        }

        // 计算标准差（SIMD 加速平方和）
        let start = i + 1 - timeperiod;
        let window = &input[start..=i];
        let sum_sq = crate::simd::sum_sq_diff(window, ma_val);
        let stddev = (sum_sq / timeperiod as f64).sqrt();

        upper[i] = ma_val + nbdevup * stddev;
        lower[i] = ma_val - nbdevdn * stddev;
    }

    Ok((upper, ma, lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbands_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (upper, middle, lower) = bbands(&input, 5, 2.0, 2.0, MaType::Sma).unwrap();

        assert!(upper[3].is_nan());
        assert!(!upper[4].is_nan());
        assert!(!middle[4].is_nan());
        assert!(!lower[4].is_nan());

        // middle[4] = SMA(1..5) = 3.0
        assert!((middle[4] - 3.0).abs() < 1e-10);
        // upper > middle > lower
        assert!(upper[4] > middle[4]);
        assert!(middle[4] > lower[4]);
    }
}
