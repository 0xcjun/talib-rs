use crate::error::{TaError, TaResult};

/// Kaufman Adaptive Moving Average (KAMA) — O(n) 滑动窗口算法
///
/// volatility = Σ|Δp| 使用滑动窗口维护，每步 O(1)。
/// lookback = timeperiod
pub fn kama(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    let lookback = timeperiod;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];

    let fast_sc = 2.0 / (2.0 + 1.0);
    let slow_sc = 2.0 / (30.0 + 1.0);

    // 预计算逐步变动的绝对值
    let mut abs_delta = vec![0.0; len];
    for j in 1..len {
        abs_delta[j] = (input[j] - input[j - 1]).abs();
    }

    // 初始波动率: sum(abs_delta[lookback-period+2 .. lookback+1])
    // 即从 index (lookback+1-timeperiod+1) 到 (lookback) 的 abs_delta
    // = abs_delta[(lookback - timeperiod + 2)..=lookback]
    // 但 lookback = timeperiod，所以 = abs_delta[2..=timeperiod]
    let mut volatility: f64 = abs_delta[1..=lookback].iter().sum();

    let mut prev_kama = input[lookback];
    output[lookback] = prev_kama;

    for i in (lookback + 1)..len {
        // O(1) 滑动波动率:
        // 新增 abs_delta[i], 移除 abs_delta[i - timeperiod]
        // 窗口: abs_delta[(i-timeperiod+1)..=i]
        volatility += abs_delta[i] - abs_delta[i - timeperiod];

        let direction = (input[i] - input[i - timeperiod]).abs();

        let er = if volatility > 0.0 {
            direction / volatility
        } else {
            0.0
        };

        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

        let kama_val = prev_kama + sc * (input[i] - prev_kama);
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
        assert!(result[9].is_nan());
        assert!(!result[10].is_nan());
    }
}
