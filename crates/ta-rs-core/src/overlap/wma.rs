use crate::error::{TaError, TaResult};

/// Weighted Moving Average (WMA)
///
/// 权重为线性递增: 最近的数据权重最高。
/// WMA = Σ(weight_i * price_i) / Σ(weight_i)
/// 其中 weight_i = i (1, 2, ..., timeperiod)
/// lookback = timeperiod - 1
pub fn wma(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
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
    let divider = (timeperiod * (timeperiod + 1)) as f64 / 2.0;

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let mut weighted_sum = 0.0;
        for (w, j) in (start..=i).enumerate() {
            weighted_sum += input[j] * (w + 1) as f64;
        }
        output[i] = weighted_sum / divider;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wma_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wma(&input, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // WMA[2] = (1*1 + 2*2 + 3*3) / 6 = 14/6 ≈ 2.333
        let expected = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / 6.0;
        assert!((result[2] - expected).abs() < 1e-10);
    }
}
