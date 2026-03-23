use crate::error::{TaError, TaResult};
use crate::ma_type::{compute_ma, MaType};

/// Bollinger Bands (BBANDS) — 对 SMA 情况使用 O(n) 单遍计算
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

    if matype == MaType::Sma {
        // SMA 优化路径: 单遍滑动窗口同时计算 SMA 和 stddev
        return bbands_sma(input, timeperiod, nbdevup, nbdevdn);
    }

    // 非 SMA 路径: 计算 MA 后按窗口计算 stddev (需要用 MA 值作为中心)
    let ma = compute_ma(input, timeperiod, matype)?;

    let lookback = timeperiod - 1;
    let mut upper = vec![0.0_f64; len];
    upper[..lookback].fill(f64::NAN);
    let mut lower = vec![0.0_f64; len];
    lower[..lookback].fill(f64::NAN);

    for i in lookback..len {
        let ma_val = ma[i];
        if ma_val.is_nan() {
            continue;
        }

        let start = i + 1 - timeperiod;
        let window = &input[start..=i];
        let sum_sq = crate::simd::sum_sq_diff(window, ma_val);
        let stddev = (sum_sq / timeperiod as f64).sqrt();

        upper[i] = ma_val + nbdevup * stddev;
        lower[i] = ma_val - nbdevdn * stddev;
    }

    Ok((upper, ma, lower))
}

/// SMA 专用 BBANDS: O(n) 滑动窗口同时维护 sum 和 sum_sq
fn bbands_sma(
    input: &[f64],
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
) -> TaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = input.len();
    let lookback = timeperiod - 1;
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let mut upper = vec![0.0_f64; len];
    upper[..lookback].fill(f64::NAN);
    let mut middle = vec![0.0_f64; len];
    middle[..lookback].fill(f64::NAN);
    let mut lower = vec![0.0_f64; len];
    lower[..lookback].fill(f64::NAN);
    let n = timeperiod as f64;
    let inv_n = 1.0 / n;

    // 初始化第一个窗口
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for j in 0..timeperiod {
        let v = input[j];
        sum += v;
        sum_sq += v * v;
    }

    // C TA-Lib 的 stddev 使用 population variance: Var = E(X^2) - E(X)^2
    // stddev = sqrt(sum_sq/n - (sum/n)^2)
    let ma_val = sum * inv_n;
    let variance = sum_sq * inv_n - ma_val * ma_val;
    let stddev = variance.max(0.0).sqrt();
    middle[lookback] = ma_val;
    upper[lookback] = ma_val + nbdevup * stddev;
    lower[lookback] = ma_val - nbdevdn * stddev;

    // 滑动
    for i in timeperiod..len {
        unsafe {
            let old = *input.get_unchecked(i - timeperiod);
            let new = *input.get_unchecked(i);
            sum += new - old;
            sum_sq += new * new - old * old;
            let ma_val = sum * inv_n;
            let variance = sum_sq * inv_n - ma_val * ma_val;
            let stddev = variance.max(0.0).sqrt();
            *middle.get_unchecked_mut(i) = ma_val;
            *upper.get_unchecked_mut(i) = ma_val + nbdevup * stddev;
            *lower.get_unchecked_mut(i) = ma_val - nbdevdn * stddev;
        }
    }

    Ok((upper, middle, lower))
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
