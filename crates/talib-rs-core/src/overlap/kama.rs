use crate::error::{TaError, TaResult};

/// Kaufman Adaptive Moving Average (KAMA) — O(n) 滑动窗口算法
///
/// C TA-Lib compatible: lookback = timeperiod, seed = close[timeperiod].
/// volatility = Σ|Δp| 使用滑动窗口维护，每步 O(1)。
pub fn kama(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    let lookback = timeperiod; // C TA-Lib: lookback = period, first output at index period
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    let fast_sc = 2.0 / (2.0 + 1.0);
    let slow_sc = 2.0 / (30.0 + 1.0);
    let sc_diff = fast_sc - slow_sc;

    // C TA-Lib: internal seed = close[period-1], first output at index `period`
    let mut prev_kama: f64 = input[timeperiod - 1];

    // 初始化波动率: sum of |delta| for indices 1..=timeperiod
    let mut volatility = 0.0_f64;
    for j in 1..=timeperiod {
        volatility += (input[j] - input[j - 1]).abs();
    }

    // First output at i = lookback (no volatility update needed)
    {
        let i = lookback;
        let direction = (input[i] - input[i - timeperiod]).abs();
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc_raw = er * sc_diff + slow_sc;
        let sc = sc_raw * sc_raw;
        let close_i = input[i];
        let kama_val = prev_kama + sc * (close_i - prev_kama);
        output[i] = kama_val;
        prev_kama = kama_val;
    }

    // Remaining outputs with sliding volatility update
    for i in (lookback + 1)..len {
        volatility += (input[i] - input[i - 1]).abs()
            - (input[i - timeperiod]
                - input[i - timeperiod - 1])
                .abs();

        let direction = (input[i] - input[i - timeperiod]).abs();
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc_raw = er * sc_diff + slow_sc;
        let sc = sc_raw * sc_raw;
        let close_i = input[i];
        let kama_val = prev_kama + sc * (close_i - prev_kama);
        output[i] = kama_val;
        prev_kama = kama_val;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kama_basic() {
        let input: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = kama(&input, 10).unwrap();
        // lookback = 10, first valid at index 10, index 9 is NaN
        assert!(result[9].is_nan());
        assert!(!result[10].is_nan());
    }
}
