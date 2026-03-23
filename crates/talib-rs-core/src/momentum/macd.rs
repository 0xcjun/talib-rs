use crate::error::{TaError, TaResult};
use crate::ma_type::{compute_ma, MaType};

/// MACD (Moving Average Convergence/Divergence)
///
/// 返回 (macd_line, signal_line, histogram)
/// macd_line = EMA(fast) - EMA(slow)
/// signal = EMA(macd_line, signalperiod)
/// histogram = macd_line - signal
///
/// C TA-Lib 兼容：MACD 内部的 fast/slow EMA 与独立 EMA 不同。
/// - slow EMA seed = SMA(close[0..slowperiod])
/// - fast EMA seed = SMA(close[slowperiod-fastperiod..slowperiod])
/// - 两条 EMA 从 bar slowperiod 开始递推
/// - MACD line 第一个值 (bar slowperiod-1) = fast_seed - slow_seed
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

    let k_fast = 2.0 / (fp as f64 + 1.0);
    let k_slow = 2.0 / (sp as f64 + 1.0);
    let k_signal = 2.0 / (signalperiod as f64 + 1.0);

    // C TA-Lib MACD 内部 EMA 计算：
    // slow seed = SMA(close[0..sp]), fast seed = SMA(close[sp-fp..sp])
    let slow_seed: f64 = input[..sp].iter().sum::<f64>() / sp as f64;
    let fast_seed: f64 = input[sp - fp..sp].iter().sum::<f64>() / fp as f64;

    // MACD line: 第一个值 (对应 bar sp-1) = fast_seed - slow_seed
    // 后续从 bar sp 开始递推
    let mut macd_values = Vec::with_capacity(len - sp + 1);
    macd_values.push(fast_seed - slow_seed);

    let mut slow_ema = slow_seed;
    let mut fast_ema = fast_seed;
    for i in sp..len {
        slow_ema = input[i].mul_add(k_slow, slow_ema * (1.0 - k_slow));
        fast_ema = input[i].mul_add(k_fast, fast_ema * (1.0 - k_fast));
        macd_values.push(fast_ema - slow_ema);
    }

    // Signal line = EMA(macd_values, signalperiod)
    // seed = SMA(macd_values[0..signalperiod])
    let signal_seed: f64 =
        macd_values[..signalperiod].iter().sum::<f64>() / signalperiod as f64;

    // 构建输出
    let out_start = sp - 1 + signalperiod - 1; // = lookback
    let mut macd_line = vec![0.0_f64; len];
    macd_line[..out_start].fill(f64::NAN);
    let mut signal_line = vec![0.0_f64; len];
    signal_line[..out_start].fill(f64::NAN);
    let mut histogram = vec![0.0_f64; len];
    histogram[..out_start].fill(f64::NAN);

    // signal 第一个值对应 macd_values[signalperiod-1]，即 bar out_start
    let mut signal_ema = signal_seed;
    let macd_at_out_start = macd_values[signalperiod - 1];
    macd_line[out_start] = macd_at_out_start;
    signal_line[out_start] = signal_seed;
    histogram[out_start] = macd_at_out_start - signal_seed;

    for i in signalperiod..macd_values.len() {
        let bar = sp - 1 + i;
        signal_ema = macd_values[i].mul_add(k_signal, signal_ema * (1.0 - k_signal));
        macd_line[bar] = macd_values[i];
        signal_line[bar] = signal_ema;
        histogram[bar] = macd_values[i] - signal_ema;
    }

    Ok((macd_line, signal_line, histogram))
}

/// MACD with controllable MA Type
///
/// Fast path for all-EMA: single-pass scalar (same as macd(), reuses its structure).
/// Generic path: 3× compute_ma with contiguous NaN detection.
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

    let valid_start = fast_ma
        .iter()
        .zip(slow_ma.iter())
        .position(|(f, s)| !f.is_nan() && !s.is_nan())
        .unwrap_or(len);

    let macd_valid: Vec<f64> = fast_ma[valid_start..]
        .iter()
        .zip(slow_ma[valid_start..].iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_ma = compute_ma(&macd_valid, signalperiod, signalmatype)?;

    let sig_start = signal_ma
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(signal_ma.len());

    let mut macd_line = vec![f64::NAN; len];
    let mut signal_line = vec![f64::NAN; len];
    let mut histogram = vec![f64::NAN; len];

    for j in sig_start..signal_ma.len() {
        let orig = valid_start + j;
        if orig < len {
            macd_line[orig] = macd_valid[j];
            signal_line[orig] = signal_ma[j];
            histogram[orig] = macd_valid[j] - signal_ma[j];
        }
    }

    Ok((macd_line, signal_line, histogram))
}

/// MACD Fix (12, 26 固定参数)
///
/// C TA-Lib MACDFIX 使用固定 k 值（k_fast=0.15, k_slow=0.075），
/// 且采用与 MACD 相同的对齐式 EMA：
/// - slow seed = SMA(close[0..26])
/// - fast seed = SMA(close[14..26])  (从 slow 窗口末尾取 fast 长度)
/// - 两条 EMA 都从 bar 26 开始递推
pub fn macd_fix(input: &[f64], signalperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = input.len();
    let fp = 12usize;
    let sp = 26usize;
    let k_fast = 0.15;  // C TA-Lib fixed k for period 12
    let k_slow = 0.075; // C TA-Lib fixed k for period 26

    let lookback = sp - 1 + signalperiod - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let k_signal = 2.0 / (signalperiod as f64 + 1.0);

    // 对齐式 EMA seed (same as MACD internal):
    let slow_seed: f64 = input[..sp].iter().sum::<f64>() / sp as f64;
    let fast_seed: f64 = input[sp - fp..sp].iter().sum::<f64>() / fp as f64;

    let mut macd_values = Vec::with_capacity(len - sp + 1);
    macd_values.push(fast_seed - slow_seed);

    let mut slow_ema = slow_seed;
    let mut fast_ema = fast_seed;
    for i in sp..len {
        slow_ema = input[i].mul_add(k_slow, slow_ema * (1.0 - k_slow));
        fast_ema = input[i].mul_add(k_fast, fast_ema * (1.0 - k_fast));
        macd_values.push(fast_ema - slow_ema);
    }

    let signal_seed: f64 =
        macd_values[..signalperiod].iter().sum::<f64>() / signalperiod as f64;

    let out_start = sp - 1 + signalperiod - 1;
    let mut macd_line = vec![0.0_f64; len];
    macd_line[..out_start].fill(f64::NAN);
    let mut signal_line = vec![0.0_f64; len];
    signal_line[..out_start].fill(f64::NAN);
    let mut histogram = vec![0.0_f64; len];
    histogram[..out_start].fill(f64::NAN);

    let mut sig_ema = signal_seed;
    let macd_at_out = macd_values[signalperiod - 1];
    macd_line[out_start] = macd_at_out;
    signal_line[out_start] = signal_seed;
    histogram[out_start] = macd_at_out - signal_seed;

    for i in signalperiod..macd_values.len() {
        let bar = sp - 1 + i;
        sig_ema = macd_values[i].mul_add(k_signal, sig_ema * (1.0 - k_signal));
        macd_line[bar] = macd_values[i];
        signal_line[bar] = sig_ema;
        histogram[bar] = macd_values[i] - sig_ema;
    }

    Ok((macd_line, signal_line, histogram))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd_basic() {
        let input: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let (macd_line, signal, hist) = macd(&input, 12, 26, 9).unwrap();
        // C TA-Lib: all three outputs start at index slowperiod-1 + signalperiod-1 = 25+8 = 33
        assert!(macd_line[32].is_nan());
        assert!(!macd_line[33].is_nan());
        assert!(signal[32].is_nan());
        assert!(!signal[33].is_nan());
        assert!(hist[32].is_nan());
        assert!(!hist[33].is_nan());
    }
}
