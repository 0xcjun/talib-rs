use crate::error::{TaError, TaResult};
use crate::ma_type::{compute_ma, MaType};

/// Stochastic Oscillator (STOCH)
///
/// 返回 (slowk, slowd)
/// FastK = 100 * (close - lowest_low) / (highest_high - lowest_low)
/// SlowK = MA(FastK, slowk_period)
/// SlowD = MA(SlowK, slowd_period)
pub fn stoch(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    slowk_period: usize,
    slowk_matype: MaType,
    slowd_period: usize,
    slowd_matype: MaType,
) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if fastk_period < 1 || slowk_period < 1 || slowd_period < 1 {
        return Err(TaError::InvalidParameter {
            name: "periods",
            value: format!("{}/{}/{}", fastk_period, slowk_period, slowd_period),
            reason: "all periods must be >= 1",
        });
    }

    let lookback = fastk_period - 1 + slowk_period - 1 + slowd_period - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    // 计算 FastK
    let mut fastk = Vec::with_capacity(len - (fastk_period - 1));
    for i in (fastk_period - 1)..len {
        let start = i + 1 - fastk_period;
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        for j in start..=i {
            if high[j] > hh {
                hh = high[j];
            }
            if low[j] < ll {
                ll = low[j];
            }
        }
        let range = hh - ll;
        if range > 0.0 {
            fastk.push(100.0 * (close[i] - ll) / range);
        } else {
            fastk.push(50.0); // TA-Lib 当 range=0 时返回 50
        }
    }

    // SlowK = MA(FastK)
    let slowk_arr = compute_ma(&fastk, slowk_period, slowk_matype)?;
    let slowk_valid: Vec<f64> = slowk_arr.iter().copied().filter(|v| !v.is_nan()).collect();

    // SlowD = MA(SlowK)
    let slowd_arr = compute_ma(&slowk_valid, slowd_period, slowd_matype)?;

    // 映射到原始长度
    let mut slowk_out = vec![f64::NAN; len];
    let mut slowd_out = vec![f64::NAN; len];

    let k_start = fastk_period - 1 + slowk_period - 1;
    for (j, i) in (k_start..len).enumerate() {
        if j < slowk_valid.len() {
            slowk_out[i] = slowk_valid[j];
        }
    }

    let d_start = k_start + slowd_period - 1;
    let mut d_idx = 0;
    for i in d_start..len {
        if d_idx < slowd_arr.len() && !slowd_arr[d_idx].is_nan() {
            slowd_out[i] = slowd_arr[d_idx];
        }
        d_idx += 1;
    }

    Ok((slowk_out, slowd_out))
}

/// Fast Stochastic (STOCHF)
///
/// 返回 (fastk, fastd)
pub fn stochf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: MaType,
) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }

    let lookback = fastk_period - 1 + fastd_period - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut fastk_values = Vec::with_capacity(len - (fastk_period - 1));
    for i in (fastk_period - 1)..len {
        let start = i + 1 - fastk_period;
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        for j in start..=i {
            if high[j] > hh {
                hh = high[j];
            }
            if low[j] < ll {
                ll = low[j];
            }
        }
        let range = hh - ll;
        if range > 0.0 {
            fastk_values.push(100.0 * (close[i] - ll) / range);
        } else {
            fastk_values.push(50.0);
        }
    }

    let mut fastk_out = vec![f64::NAN; len];
    for (j, i) in ((fastk_period - 1)..len).enumerate() {
        fastk_out[i] = fastk_values[j];
    }

    let fastd_arr = compute_ma(&fastk_values, fastd_period, fastd_matype)?;
    let mut fastd_out = vec![f64::NAN; len];
    let d_start = fastk_period - 1 + fastd_period - 1;
    let mut d_idx = fastd_period - 1;
    for i in d_start..len {
        if d_idx < fastd_arr.len() {
            fastd_out[i] = fastd_arr[d_idx];
        }
        d_idx += 1;
    }

    Ok((fastk_out, fastd_out))
}

/// Stochastic RSI (STOCHRSI)
///
/// 返回 (fastk, fastd)
pub fn stochrsi(
    input: &[f64],
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: MaType,
) -> TaResult<(Vec<f64>, Vec<f64>)> {
    // 先计算 RSI
    let rsi_values = crate::momentum::rsi::rsi(input, timeperiod)?;

    // 收集非 NaN 的 RSI 值
    let rsi_valid: Vec<f64> = rsi_values.iter().copied().filter(|v| !v.is_nan()).collect();

    if rsi_valid.len() <= fastk_period {
        return Err(TaError::InsufficientData {
            need: timeperiod + fastk_period + 1,
            got: input.len(),
        });
    }

    // 对 RSI 应用 Stochastic
    let rsi_len = rsi_valid.len();
    let mut fastk_values = Vec::new();
    for i in (fastk_period - 1)..rsi_len {
        let start = i + 1 - fastk_period;
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        for j in start..=i {
            if rsi_valid[j] > hh {
                hh = rsi_valid[j];
            }
            if rsi_valid[j] < ll {
                ll = rsi_valid[j];
            }
        }
        let range = hh - ll;
        if range > 0.0 {
            fastk_values.push(100.0 * (rsi_valid[i] - ll) / range);
        } else {
            fastk_values.push(50.0);
        }
    }

    let fastd_arr = compute_ma(&fastk_values, fastd_period, fastd_matype)?;

    // 映射到原始长度
    let len = input.len();
    let mut fastk_out = vec![f64::NAN; len];
    let mut fastd_out = vec![f64::NAN; len];

    let k_start = timeperiod + fastk_period - 1;
    for (j, i) in (k_start..len).enumerate() {
        if j < fastk_values.len() {
            fastk_out[i] = fastk_values[j];
        }
    }

    let d_start = k_start + fastd_period - 1;
    for (j, i) in (d_start..len).enumerate() {
        let idx = fastd_period - 1 + j;
        if idx < fastd_arr.len() && !fastd_arr[idx].is_nan() {
            fastd_out[i] = fastd_arr[idx];
        }
    }

    Ok((fastk_out, fastd_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stoch_basic() {
        let high: Vec<f64> = (0..30)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0 + 2.0)
            .collect();
        let low: Vec<f64> = (0..30)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0 - 2.0)
            .collect();
        let close: Vec<f64> = (0..30)
            .map(|i| 50.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let (slowk, slowd) = stoch(&high, &low, &close, 5, 3, MaType::Sma, 3, MaType::Sma).unwrap();
        assert_eq!(slowk.len(), 30);
    }
}
