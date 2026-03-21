use crate::error::{TaError, TaResult};
use crate::overlap::ema::{ema, ema_core};
use crate::ma_type::{compute_ma, MaType};

/// MACD (Moving Average Convergence/Divergence)
///
/// 返回 (macd_line, signal_line, histogram)
/// macd_line = EMA(fast) - EMA(slow)
/// signal = EMA(macd_line, signalperiod)
/// histogram = macd_line - signal
///
/// lookback = slowperiod - 1 + signalperiod - 1
pub fn macd(
    input: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> TaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if fastperiod < 2 || slowperiod < 2 || signalperiod < 1 {
        return Err(TaError::InvalidParameter {
            name: "fastperiod/slowperiod/signalperiod",
            value: format!("{}/{}/{}", fastperiod, slowperiod, signalperiod),
            reason: "fastperiod >= 2, slowperiod >= 2, signalperiod >= 1",
        });
    }

    // 确保 slow > fast
    let (fp, sp) = if fastperiod < slowperiod {
        (fastperiod, slowperiod)
    } else {
        (slowperiod, fastperiod)
    };

    let len = input.len();
    let lookback = sp - 1 + signalperiod - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let fast_ema = ema(input, fp)?;
    let slow_ema = ema(input, sp)?;

    // MACD Line
    let macd_start = sp - 1; // slow EMA 第一个有效位置
    let mut macd_line = vec![f64::NAN; len];
    let mut macd_valid = Vec::with_capacity(len - macd_start);
    for i in macd_start..len {
        let val = fast_ema[i] - slow_ema[i];
        macd_line[i] = val;
        macd_valid.push(val);
    }

    // Signal Line = EMA(MACD Line)
    let k = 2.0 / (signalperiod as f64 + 1.0);
    let signal_on_valid = ema_core(&macd_valid, signalperiod, k)?;

    // 映射回原始长度
    let mut signal_line = vec![f64::NAN; len];
    let mut histogram = vec![f64::NAN; len];
    let signal_start = macd_start + signalperiod - 1;

    for i in 0..signal_on_valid.len() {
        let orig_idx = macd_start + i;
        if !signal_on_valid[i].is_nan() && orig_idx < len {
            signal_line[orig_idx] = signal_on_valid[i];
            histogram[orig_idx] = macd_line[orig_idx] - signal_on_valid[i];
        }
    }

    Ok((macd_line, signal_line, histogram))
}

/// MACD with controllable MA Type
pub fn macd_ext(
    input: &[f64],
    fastperiod: usize,
    fastmatype: MaType,
    slowperiod: usize,
    slowmatype: MaType,
    signalperiod: usize,
    signalmatype: MaType,
) -> TaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (fp, sp) = if fastperiod < slowperiod {
        (fastperiod, slowperiod)
    } else {
        (slowperiod, fastperiod)
    };

    let len = input.len();
    let fast_ma = compute_ma(input, fp, fastmatype)?;
    let slow_ma = compute_ma(input, sp, slowmatype)?;

    let macd_start = sp - 1;
    let mut macd_line = vec![f64::NAN; len];
    let mut macd_valid = Vec::new();
    for i in macd_start..len {
        if !fast_ma[i].is_nan() && !slow_ma[i].is_nan() {
            let val = fast_ma[i] - slow_ma[i];
            macd_line[i] = val;
            macd_valid.push(val);
        }
    }

    let signal_ma = compute_ma(&macd_valid, signalperiod, signalmatype)?;

    let mut signal_line = vec![f64::NAN; len];
    let mut histogram = vec![f64::NAN; len];

    let mut valid_idx = 0;
    for i in macd_start..len {
        if !macd_line[i].is_nan() {
            if valid_idx < signal_ma.len() && !signal_ma[valid_idx].is_nan() {
                signal_line[i] = signal_ma[valid_idx];
                histogram[i] = macd_line[i] - signal_ma[valid_idx];
            }
            valid_idx += 1;
        }
    }

    Ok((macd_line, signal_line, histogram))
}

/// MACD Fix (26, 12, 9 固定参数)
pub fn macd_fix(
    input: &[f64],
    signalperiod: usize,
) -> TaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    macd(input, 12, 26, signalperiod)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd_basic() {
        let input: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let (macd_line, signal, hist) = macd(&input, 12, 26, 9).unwrap();
        // lookback = 25 + 8 = 33
        assert!(macd_line[24].is_nan());
        assert!(!macd_line[25].is_nan());
        // signal 从 25 + 8 = 33 开始
        assert!(signal[32].is_nan());
        assert!(!signal[33].is_nan());
    }
}
